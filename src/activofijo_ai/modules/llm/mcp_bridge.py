import importlib.util
import re
import unicodedata
from datetime import datetime
from typing import Any

from activofijo_ai.core.config import Settings
from activofijo_ai.modules.llm.knowledge import (
    GUIA_RUTAS_SISTEMA,
    PALABRAS_BLOQUEO_EXTERNO,
    PALABRAS_DOMINIO_VALIDO,
    PLANTILLAS_CARTAS,
    RutaSistemaGuia,
)
from activofijo_ai.modules.llm.provider import (
    LlmErrorExternoError,
    LlmNoConfiguradoError,
    OpenAICompatibleLlmClient,
    extraer_json_objeto,
)
from activofijo_ai.modules.llm.schemas import (
    AyudaSistemaRequest,
    AyudaSistemaResponse,
    ConsultaLenguajeNaturalRequest,
    ConsultaLenguajeNaturalResponse,
    EnlaceDirectoSistema,
    FiltrosDetectadosReporte,
    RedaccionAsistidaRequest,
    RedaccionAsistidaResponse,
)
from activofijo_ai.modules.system.schemas import EstadoMcp

HERRAMIENTAS_MCP_PLANIFICADAS: list[str] = [
    "consultar_ficha_activo",
    "consultar_reglas_normativas",
    "filtrar_catalogo_reportes",
    "consultar_guia_sistema",
    "validar_integridad_texto",
]

MESES_ES = (
    "enero",
    "febrero",
    "marzo",
    "abril",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
)


def _normalizar_texto(texto: str) -> str:
    sin_tildes = "".join(
        c for c in unicodedata.normalize("NFD", texto.lower()) if unicodedata.category(c) != "Mn"
    )
    return re.sub(r"\s+", " ", sin_tildes).strip()


def _fecha_bolivia_formal() -> str:
    hoy = datetime.now()
    mes = MESES_ES[hoy.month - 1]
    return f"Santa Cruz de la Sierra, {hoy.day:02d} de {mes} de {hoy.year}"


def _formatear_bs(valor: Any) -> str:
    try:
        num = float(valor or 0)
    except (TypeError, ValueError):
        num = 0.0
    monto_es = f"{num:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"Bs. {monto_es}"


def _resolver_motor(settings: Settings) -> tuple[str, str]:
    if settings.llm_provider.lower() == "ollama":
        return "ollama_personalizado", settings.llm_model
    return "llm_remoto", settings.llm_model


def _tiene_permiso(permiso_requerido: str | None, permisos_usuario: list[str], rol: str) -> bool:
    if permiso_requerido is None:
        return True
    rol_up = rol.upper()
    if rol_up in {"ADMINISTRADOR", "SUPER_ADMIN", "ADMIN"}:
        return True
    return permiso_requerido in permisos_usuario


class McpBridgeService:
    """Puente MCP de solo lectura hacia recursos UAGRM, validación y orquestación LLM local."""

    def __init__(self, transport: Any = None) -> None:
        self._transport = transport

    def obtener_estado(self) -> EstadoMcp:
        """Indica si el SDK MCP está instalado y qué herramientas de solo lectura están registradas."""
        sdk_disponible = importlib.util.find_spec("mcp") is not None
        return EstadoMcp(
            habilitado=sdk_disponible,
            herramientas_registradas=list(HERRAMIENTAS_MCP_PLANIFICADAS),
        )

    async def asistir_redaccion(
        self,
        payload: RedaccionAsistidaRequest,
        settings: Settings,
    ) -> RedaccionAsistidaResponse:
        plantilla = PLANTILLAS_CARTAS[payload.tipo_documento]
        ctx = payload.contexto_activo or {}
        codigo_activo = str(payload.cod_activo or ctx.get("codigo") or "").strip() or None
        es_nuevo_sin_anexo = payload.tipo_documento == "SOLICITUD_NUEVO_ACTIVO"
        incluye_anexo = bool(codigo_activo and ctx and not es_nuevo_sin_anexo)

        destinatario = (payload.destinatario_nombre or plantilla.destinatario_default).strip()
        destinatario_cargo = (
            payload.destinatario_cargo or plantilla.cargo_destinatario_default
        ).strip()
        lugar_fecha = _fecha_bolivia_formal()

        # Intentar redacción vía LLM local personalizado (uagrm-activo-fijo) con anclaje MCP
        recurso_mcp = {
            "tipo_documento": payload.tipo_documento,
            "modo_acceso": payload.modo_acceso,
            "incluye_anexo_tecnico": incluye_anexo,
            "solicitante": {
                "nombre": payload.solicitante_nombre or "Responsable / Funcionario UAGRM",
                "cargo": payload.solicitante_cargo or "Funcionario Institucional",
                "unidad": payload.solicitante_unidad or "Unidad Universitaria UAGRM",
            },
            "normativa_aplicable": list(plantilla.normas),
            "recurso_activo_mcp": ctx if not es_nuevo_sin_anexo else None,
        }

        prompt_usuario = (
            "Redacta el cuerpo epistolar de una carta formal UAGRM en español de Bolivia "
            "usando EXCLUSIVAMENTE los datos del Recurso MCP adjunto.\n"
            f"Recurso MCP: {recurso_mcp}\n"
            f"Motivo o instrucción del solicitante: {payload.instruccion_usuario}\n\n"
            "Reglas de estilo:\n"
            "- NO incluyas fecha, destinatario, 'Señor(a)', 'Presente' ni firma dentro de cuerpo_parrafos (el formato Word/PDF ya los agrega).\n"
            "- Moneda oficial: Bolivianos abreviado estrictamente como 'Bs.' (NUNCA uses 'Bs.S' ni dólares).\n"
            "- Genera exactamente 3 párrafos claros y formales en cuerpo_parrafos:\n"
            "  1) Objeto de la solicitud y referencia a la unidad solicitante y normativa UAGRM aplicable.\n"
            "  2) Detalle técnico del activo (si existe codigo en Recurso MCP, cítalo exacto con su descripción y monto) o especificación del requerimiento.\n"
            "  3) Párrafo de disposición para verificación técnica/anexo y cierre cordial.\n\n"
            "Devuelve SOLO un objeto JSON con las claves exactas:\n"
            '{"referencia": string, "cuerpo_parrafos": [string, string, string], "despedida": string}'
        )

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        advertencias: list[str] = []

        if settings.env != "test" or self._transport is not None:
            try:
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": (
                                "Redactor institucional UAGRM bajo Protocolo MCP. "
                                "Usa únicamente los códigos y valores del recurso MCP; nunca inventes cifras."
                            ),
                        },
                        {"role": "user", "content": prompt_usuario},
                    ],
                    response_format={"type": "json_object"},
                )
                datos = extraer_json_objeto(res.contenido)
                if datos and isinstance(datos.get("cuerpo_parrafos"), list):
                    parrafos = [
                        re.sub(r"\bBs\.?\s*S\.?", "Bs.", str(p)).strip()
                        for p in datos["cuerpo_parrafos"]
                        if str(p).strip()
                    ]
                    if parrafos:
                        referencia = str(datos.get("referencia") or plantilla.referencia_base).strip()
                        despedida = str(
                            datos.get("despedida")
                            or "Con este motivo, saludo a usted con las consideraciones más distinguidas."
                        ).strip()

                        # Validador MCP Anti-Alucinación: asegura cita exacta del código de activo si existe
                        texto_unido = " ".join(parrafos)
                        if codigo_activo and not es_nuevo_sin_anexo and codigo_activo not in texto_unido:
                            parrafos.insert(
                                1,
                                self._construir_parrafo_respaldo_activo(ctx, codigo_activo),
                            )
                            advertencias.append(
                                "El validador MCP insertó automáticamente la referencia exacta del activo para evitar omisiones."
                            )

                        motor_id, _ = _resolver_motor(settings)
                        return RedaccionAsistidaResponse(
                            motor_utilizado=motor_id,  # type: ignore[arg-type]
                            modelo=res.modelo,
                            referencia=referencia,
                            lugar_fecha=lugar_fecha,
                            destinatario_nombre=destinatario,
                            destinatario_cargo=destinatario_cargo,
                            cuerpo_parrafos=parrafos,
                            despedida=despedida,
                            incluye_anexo_tecnico=incluye_anexo,
                            fundamento_normativo=list(plantilla.normas),
                            advertencias=advertencias,
                            tokens_usados=res.tokens_usados,
                        )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                advertencias.append(
                    "Modelo Ollama local no disponible en este instante; borrador generado mediante motor normativo determinista MCP."
                )

        # Respaldo determinista institucional 100% garantizado
        parrafos_det = self._redactar_determinista(payload, plantilla.referencia_base, ctx, codigo_activo, incluye_anexo)
        return RedaccionAsistidaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (respaldo-determinista)",
            referencia=(
                f"{plantilla.referencia_base} — {codigo_activo}"
                if codigo_activo and not es_nuevo_sin_anexo
                else plantilla.referencia_base
            ),
            lugar_fecha=lugar_fecha,
            destinatario_nombre=destinatario,
            destinatario_cargo=destinatario_cargo,
            cuerpo_parrafos=parrafos_det,
            despedida=(
                "Agradeciendo de antemano su atención favorable a la presente gestión, "
                "me despido con las consideraciones más distinguidas."
            ),
            incluye_anexo_tecnico=incluye_anexo,
            fundamento_normativo=list(plantilla.normas),
            advertencias=advertencias,
            tokens_usados=0,
        )

    def _construir_parrafo_respaldo_activo(self, ctx: dict[str, Any], codigo: str) -> str:
        desc = str(ctx.get("descripcion") or "Bien patrimonial universitario").strip()
        ubicacion = str(
            (ctx.get("asignacion") or {}).get("ubicacion", {}).get("oficina")
            or ctx.get("ubicacion")
            or "dependencia universitaria asignada"
        ).strip()
        costo = _formatear_bs(ctx.get("costo") if ctx.get("costo") is not None else ctx.get("valor"))
        estado = str(ctx.get("estado") or "ACTIVO").strip()
        return (
            f"El bien objeto de la presente solicitud corresponde al código patrimonial {codigo} "
            f"({desc}), registrado actualmente en {ubicacion}, en estado operativo {estado} "
            f"con valor contable de {costo}."
        )

    def _redactar_determinista(
        self,
        payload: RedaccionAsistidaRequest,
        _referencia: str,
        ctx: dict[str, Any],
        codigo: str | None,
        incluye_anexo: bool,
    ) -> list[str]:
        solicitante_unidad = payload.solicitante_unidad or "la unidad universitaria correspondiente"
        instruccion = payload.instruccion_usuario.strip().rstrip(".")

        if payload.tipo_documento == "SOLICITUD_NUEVO_ACTIVO":
            return [
                (
                    f"De mi mayor consideración: Por intermedio de la presente nota, en representación de "
                    f"{solicitante_unidad}, solicito respetuosamente la dotación e incorporación de nuevo equipamiento "
                    f"institucional en el marco del D.S. N° 0181 (NB-SABS), conforme al siguiente detalle operativo: "
                    f"{instruccion}."
                ),
                (
                    "El requerimiento responde a la necesidad de garantizar la continuidad de las actividades académicas "
                    "y administrativas asignadas a nuestra dependencia, optimizando las condiciones operativas del personal."
                ),
                (
                    "Una vez autorizada la adquisición o asignación correspondiente, la unidad solicitante asumirá el "
                    "registro, verificación de inventario y custodia formal mediante acta oficial del Departamento de Activos Fijos."
                ),
            ]

        introduccion_map = {
            "SOLICITUD_CORRECCION_DATOS": (
                "solicito formalmente la corrección y actualización en el padrón patrimonial de los datos técnicos "
                "o de registro asignados al bien descrito a continuación"
            ),
            "SOLICITUD_BAJA": (
                "solicito respetuosamente la evaluación técnica y desincorporación (baja patrimonial) conforme al "
                "Manual de Procedimientos de Baja de Bienes (R.R. N° 449/2013)"
            ),
            "SOLICITUD_INVENTARIO": (
                "solicito la programación de levantamiento e inventario físico verificado de los bienes asignados "
                "a nuestra dependencia conforme al D.S. N° 0181"
            ),
            "SOLICITUD_REASIGNACION": (
                "solicito formalizar la reasignación y transferencia administrativa de custodia patrimonial hacia "
                "la dependencia o funcionario receptor correspondiente"
            ),
            "SOLICITUD_REVALUO": (
                "solicito la inspección y revalorización técnica pericial al amparo del Manual UAGRM-MRVT-AF "
                "(R.R. N° 567/2018 V2) considerando que el bien conserva condiciones operativas útiles"
            ),
            "DESCARGO_INCIDENTE": (
                "elevo el presente informe de descargo oportuno en cumplimiento de los Arts. 8 y 12 del Manual "
                "R.R. N° 449/2013 y la Ley N° 1178 para fines de constancia administrativa y verificación técnica"
            ),
            "INFORME_TECNICO": (
                "remito el presente Dictamen Técnico de evaluación patrimonial sustentado en los registros del "
                "padrón institucional y la normativa universitaria vigente"
            ),
        }
        verbo = introduccion_map.get(
            payload.tipo_documento,
            "presento formalmente la siguiente solicitud ante el Departamento de Activos Fijos",
        )

        parrafos = [
            (
                f"De mi mayor consideración: En cumplimiento de la normativa patrimonial vigente de la UAGRM "
                f"y en representación de {solicitante_unidad}, {verbo}, conforme al siguiente fundamento: {instruccion}."
            )
        ]

        if codigo and ctx:
            parrafos.append(self._construir_parrafo_respaldo_activo(ctx, codigo))
        elif codigo:
            parrafos.append(
                f"El trámite comprende al bien patrimonial identificado con el código {codigo}."
            )

        cierre = (
            "Se adjunta como Anexo en la página siguiente el Informe Técnico oficial extraído del padrón institucional "
            "para su verificación y trámite reglamentario."
            if incluye_anexo
            else "Quedo a entera disposición para coordinar la verificación documental o en sitio que Jefatura estime pertinente."
        )
        parrafos.append(cierre)
        return parrafos

    async def interpretar_consulta_reporte(
        self,
        payload: ConsultaLenguajeNaturalRequest,
        settings: Settings,
    ) -> ConsultaLenguajeNaturalResponse:
        filtros_det = self._extraer_filtros_deterministas(payload.consulta_usuario)
        lote = payload.resumen_lote
        resumen_ej: str | None = None
        hallazgos: list[str] = []
        advertencias: list[str] = []

        if lote is not None:
            resumen_ej, hallazgos = self._construir_resumen_ejecutivo_lote(lote)

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        motor_usado = "respaldo_determinista"
        modelo_usado = f"{settings.llm_model} (respaldo-determinista)"
        tokens = 0

        if (payload.consulta_usuario.strip() or lote is not None) and (
            settings.env != "test" or self._transport is not None
        ):
            try:
                prompt = (
                    "Eres un intérprete de consultas MCP y analista patrimonial de la UAGRM.\n"
                    f"Módulo: {payload.contexto_modulo}\n"
                    f"Consulta en lenguaje natural: {payload.consulta_usuario or 'Sin filtro adicional'}\n"
                    f"Resumen real del lote (JSON): {lote.model_dump() if lote else None}\n\n"
                    "Devuelve SOLO JSON con estas claves:\n"
                    '{"filtros": {"search": str|null, "estado": "ACTIVO"|"ASIGNADO"|"BAJA"|null, '
                    '"condicion": "BUENO"|"REGULAR"|"MALO"|null, "grupo_contable": str|null, '
                    '"ubicacion": str|null, "valor_min": float|null, "valor_max": float|null, "gestion": int|null}, '
                    '"explicacion": str, "resumen_ejecutivo": str|null, "hallazgos_clave": [str]}'
                )
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": "Traductor MCP de consultas a JSON estricto para reportes patrimoniales UAGRM.",
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = extraer_json_objeto(res.contenido)
                if parsed:
                    raw_f = parsed.get("filtros")
                    if isinstance(raw_f, dict):
                        filtros_det = self._combinar_filtros(filtros_det, raw_f)
                    if isinstance(parsed.get("resumen_ejecutivo"), str) and parsed["resumen_ejecutivo"].strip():
                        # Validar que mencione cifras reales del lote para prevenir alucinaciones
                        texto_res = parsed["resumen_ejecutivo"].strip()
                        if lote is None or str(lote.total_registros) in texto_res:
                            resumen_ej = texto_res
                    if isinstance(parsed.get("hallazgos_clave"), list) and parsed["hallazgos_clave"]:
                        hallazgos = [str(h).strip() for h in parsed["hallazgos_clave"][:4] if str(h).strip()]
                    explicacion = str(parsed.get("explicacion") or "").strip() or self._explicar_filtros(filtros_det)
                    motor_id, _ = _resolver_motor(settings)
                    return ConsultaLenguajeNaturalResponse(
                        motor_utilizado=motor_id,  # type: ignore[arg-type]
                        modelo=res.modelo,
                        filtros_detectados=filtros_det,
                        explicacion=explicacion,
                        resumen_ejecutivo=resumen_ej,
                        hallazgos_clave=hallazgos,
                        advertencias=advertencias,
                        tokens_usados=res.tokens_usados,
                    )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                advertencias.append("Intérprete ejecutado en modo determinista local MCP.")

        return ConsultaLenguajeNaturalResponse(
            motor_utilizado=motor_usado,  # type: ignore[arg-type]
            modelo=modelo_usado,
            filtros_detectados=filtros_det,
            explicacion=self._explicar_filtros(filtros_det),
            resumen_ejecutivo=resumen_ej,
            hallazgos_clave=hallazgos,
            advertencias=advertencias,
            tokens_usados=tokens,
        )

    def _extraer_filtros_deterministas(self, consulta: str) -> FiltrosDetectadosReporte:
        norm = _normalizar_texto(consulta)
        if not norm:
            return FiltrosDetectadosReporte()

        estado: str | None = None
        if "baja" in norm or "desincorporad" in norm:
            estado = "BAJA"
        elif "asignad" in norm or "custodia" in norm:
            estado = "ASIGNADO"
        elif "activ" in norm and "activo fijo" not in norm:
            estado = "ACTIVO"

        condicion: str | None = None
        if "malo" in norm or "inutilizad" in norm or "deteriorad" in norm or "dañad" in norm:
            condicion = "MALO"
        elif "regular" in norm:
            condicion = "REGULAR"
        elif "bueno" in norm or "optimo" in norm:
            condicion = "BUENO"

        grupo: str | None = None
        if any(p in norm for p in ("comput", "laptop", "pc", "servidor", "impresora", "monitor")):
            grupo = "EQUIPOS DE COMPUTACIÓN"
        elif any(p in norm for p in ("muebl", "escritorio", "silla", "estante", "mesa")):
            grupo = "MUEBLES Y ENSERES"
        elif any(p in norm for p in ("vehic", "camioneta", "auto", "bus")):
            grupo = "VEHÍCULOS"
        elif any(p in norm for p in ("laboratorio", "microscopio", "equipo medico")):
            grupo = "EQUIPOS DE LABORATORIO"

        ubicacion: str | None = None
        for sigla in ("ficct", "decanato", "biblioteca", "rectorado", "auditorio", "duaf", "postgrado"):
            if sigla in norm:
                ubicacion = sigla.upper()
                break

        valor_min: float | None = None
        valor_max: float | None = None
        m_min = re.search(r"(?:mayor(?:es)?\s*(?:a|que|de)|desd?e\s*bs\.?|>=?)\s*(\d+(?:[.,]\d+)?)", norm)
        if m_min:
            valor_min = float(m_min.group(1).replace(",", "."))
        m_max = re.search(r"(?:menor(?:es)?\s*(?:a|que|de)|hasta\s*bs\.?|<=?)\s*(\d+(?:[.,]\d+)?)", norm)
        if m_max:
            valor_max = float(m_max.group(1).replace(",", "."))

        gestion: int | None = None
        m_anio = re.search(r"\b(20\d{2})\b", norm)
        if m_anio:
            gestion = int(m_anio.group(1))

        # Si hay código patrimonial explícito (ej. U04000018 o 4300-01-0001)
        search: str | None = None
        m_cod = re.search(r"\b([a-z]\d{6,10}|\d{4}-\d{2}-\d{3,6})\b", consulta, flags=re.IGNORECASE)
        if m_cod:
            search = m_cod.group(1).upper()

        return FiltrosDetectadosReporte(
            search=search,
            estado=estado,
            condicion=condicion,
            grupo_contable=grupo,
            ubicacion=ubicacion,
            valor_min=valor_min,
            valor_max=valor_max,
            gestion=gestion,
        )

    def _combinar_filtros(
        self, base: FiltrosDetectadosReporte, raw: dict[str, Any]
    ) -> FiltrosDetectadosReporte:
        return FiltrosDetectadosReporte(
            search=str(raw["search"]).strip() if raw.get("search") else base.search,
            estado=str(raw["estado"]).upper() if raw.get("estado") in ("ACTIVO", "ASIGNADO", "BAJA") else base.estado,
            condicion=str(raw["condicion"]).upper() if raw.get("condicion") in ("BUENO", "REGULAR", "MALO") else base.condicion,
            grupo_contable=str(raw["grupo_contable"]).strip() if raw.get("grupo_contable") else base.grupo_contable,
            ubicacion=str(raw["ubicacion"]).strip() if raw.get("ubicacion") else base.ubicacion,
            valor_min=float(raw["valor_min"]) if isinstance(raw.get("valor_min"), (int, float)) else base.valor_min,
            valor_max=float(raw["valor_max"]) if isinstance(raw.get("valor_max"), (int, float)) else base.valor_max,
            gestion=int(raw["gestion"]) if isinstance(raw.get("gestion"), int) else base.gestion,
        )

    def _explicar_filtros(self, f: FiltrosDetectadosReporte) -> str:
        partes: list[str] = []
        if f.search:
            partes.append(f"código/texto '{f.search}'")
        if f.grupo_contable:
            partes.append(f"grupo '{f.grupo_contable}'")
        if f.estado:
            partes.append(f"estado {f.estado}")
        if f.condicion:
            partes.append(f"condición física {f.condicion}")
        if f.ubicacion:
            partes.append(f"unidad '{f.ubicacion}'")
        if f.valor_min is not None:
            partes.append(f"valor >= Bs. {f.valor_min:,.2f}")
        if f.valor_max is not None:
            partes.append(f"valor <= Bs. {f.valor_max:,.2f}")
        if f.gestion is not None:
            partes.append(f"gestión {f.gestion}")

        if not partes:
            return "Se analizará el lote actual completo sin restricciones adicionales."
        return "Filtros MCP interpretados: " + ", ".join(partes) + "."

    def _construir_resumen_ejecutivo_lote(self, lote: Any) -> tuple[str, list[str]]:
        total = int(lote.total_registros)
        monto = _formatear_bs(lote.valor_total_bs)
        estados_texto = ", ".join(f"{k}: {v}" for k, v in (lote.estados or {}).items())
        resumen = (
            f"El reporte consolidado «{lote.titulo_reporte}» comprende {total} registro(s) auditado(s) "
            f"con un valor patrimonial acumulado de {monto}."
            + (f" Distribución por estado: {estados_texto}." if estados_texto else "")
        )
        hallazgos: list[str] = [
            f"Cobertura total analizada: {total} registro(s) por un monto acumulado de {monto}.",
        ]
        for grp in (lote.grupos_top or [])[:2]:
            nombre = grp.get("nombre") or "Grupo principal"
            cnt = grp.get("cantidad") or 0
            hallazgos.append(f"Concentración destacada en {nombre}: {cnt} registro(s).")
        if lote.criterios_aplicados:
            hallazgos.append("Criterios activos: " + " | ".join(lote.criterios_aplicados[:3]))
        return resumen, hallazgos

    def _es_pregunta_fuera_de_ambito(self, pregunta_norm: str) -> bool:
        if any(p in pregunta_norm for p in PALABRAS_BLOQUEO_EXTERNO):
            return True
        if len(pregunta_norm) <= 3:
            return False
        tiene_termino_dominio = any(p in pregunta_norm for p in PALABRAS_DOMINIO_VALIDO)
        return not tiene_termino_dominio and len(pregunta_norm.split()) >= 5

    def _rutas_permitidas(self, payload: AyudaSistemaRequest) -> list[RutaSistemaGuia]:
        return [
            r
            for r in GUIA_RUTAS_SISTEMA
            if _tiene_permiso(r.permiso_requerido, payload.permisos_usuario, payload.rol_usuario)
        ]

    async def responder_ayuda_sistema(
        self,
        payload: AyudaSistemaRequest,
        settings: Settings,
    ) -> AyudaSistemaResponse:
        pregunta_norm = _normalizar_texto(payload.pregunta)
        rutas_disponibles = self._rutas_permitidas(payload)

        if self._es_pregunta_fuera_de_ambito(pregunta_norm):
            return AyudaSistemaResponse(
                motor_utilizado="respaldo_determinista",
                modelo=settings.llm_model,
                respuesta=(
                    "Soy el Asistente Especializado del Sistema de Activos Fijos de la UAGRM. "
                    "Por política institucional solo respondo consultas sobre el manejo de esta plataforma, "
                    "sus formularios, actas, trazabilidad o normativas vigentes."
                ),
                pasos=[
                    "Pregúntame dónde encontrar un formulario (ej. baja, asignación o corrección).",
                    "Consulta cómo funciona el Motor Normativo, Random Forest o cómo cambiar tu contraseña y 2FA.",
                ],
                enlaces_directos=[],
                fuera_de_ambito=True,
                tokens_usados=0,
            )

        # Seleccionar enlaces relevantes según palabras clave y permisos RBAC reales
        coincidencias: list[tuple[int, RutaSistemaGuia]] = []
        for ruta in rutas_disponibles:
            puntaje = sum(1 for kw in ruta.palabras_clave if kw in pregunta_norm)
            if puntaje > 0:
                coincidencias.append((puntaje, ruta))
        coincidencias.sort(key=lambda item: item[0], reverse=True)
        rutas_sugeridas = [item[1] for item in coincidencias[:3]]

        if "baja" in pregunta_norm and not any(r.ruta == "/formularios/baja" for r in rutas_disponibles):
            doc_ruta = next((r for r in rutas_disponibles if r.ruta == "/gestion-documental"), None)
            if doc_ruta and all(r.ruta != "/gestion-documental" for r in rutas_sugeridas):
                rutas_sugeridas.insert(0, doc_ruta)
                rutas_sugeridas = rutas_sugeridas[:3]

        enlaces = [
            EnlaceDirectoSistema(titulo=r.titulo, ruta=r.ruta, descripcion=r.descripcion)
            for r in rutas_sugeridas
        ]

        # Respuesta determinista de base (alta precisión y siempre disponible)
        respuesta_det, pasos_det = self._construir_respuesta_ayuda_determinista(
            pregunta_norm, rutas_sugeridas, payload
        )

        llm = OpenAICompatibleLlmClient(settings, transport=self._transport)
        if settings.env != "test" or self._transport is not None:
            try:
                catalogo_permitido = [
                    {"titulo": r.titulo, "ruta": r.ruta, "guia": r.guia_breve}
                    for r in (rutas_sugeridas or rutas_disponibles[:6])
                ]
                prompt = (
                    f"Rol del usuario: {payload.rol_usuario} | Ruta actual: {payload.ruta_actual}\n"
                    f"Módulos permitidos para este usuario (MCP): {catalogo_permitido}\n"
                    f"Pregunta del usuario: {payload.pregunta}\n\n"
                    "Responde en español claro y conciso guiando al usuario dentro del sistema UAGRM. "
                    "Devuelve SOLO JSON: "
                    '{"respuesta": string, "pasos": [string, string], "fuera_de_ambito": false}'
                )
                res = await llm.generar_chat_completado(
                    mensajes=[
                        {
                            "role": "system",
                            "content": (
                                "Asistente guiado UAGRM. Nunca inventes rutas ni funciones inexistentes; "
                                "usa solo los módulos permitidos recibidos."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ],
                    response_format={"type": "json_object"},
                )
                parsed = extraer_json_objeto(res.contenido)
                if parsed and isinstance(parsed.get("respuesta"), str) and parsed["respuesta"].strip():
                    pasos_llm = [
                        str(p).strip()
                        for p in (parsed.get("pasos") or [])
                        if str(p).strip()
                    ][:4]
                    motor_id, _ = _resolver_motor(settings)
                    return AyudaSistemaResponse(
                        motor_utilizado=motor_id,  # type: ignore[arg-type]
                        modelo=res.modelo,
                        respuesta=parsed["respuesta"].strip(),
                        pasos=pasos_llm or pasos_det,
                        enlaces_directos=enlaces,
                        fuera_de_ambito=bool(parsed.get("fuera_de_ambito", False)),
                        tokens_usados=res.tokens_usados,
                    )
            except (LlmNoConfiguradoError, LlmErrorExternoError):
                pass

        return AyudaSistemaResponse(
            motor_utilizado="respaldo_determinista",
            modelo=f"{settings.llm_model} (guia-mcp)",
            respuesta=respuesta_det,
            pasos=pasos_det,
            enlaces_directos=enlaces,
            fuera_de_ambito=False,
            tokens_usados=0,
        )

    def _construir_respuesta_ayuda_determinista(
        self,
        pregunta_norm: str,
        rutas_sugeridas: list[RutaSistemaGuia],
        payload: AyudaSistemaRequest,
    ) -> tuple[str, list[str]]:
        if any(p in pregunta_norm for p in ("contraseña", "contrasena", "password", "2fa", "totp", "autenticador")):
            return (
                "Puede cambiar su contraseña institucional y gestionar su verificación en dos pasos (2FA TOTP) desde Seguridad de Cuenta.",
                [
                    "Haga clic en el botón 'Seguridad de Cuenta' de abajo (o en su perfil al pie del menú lateral).",
                    "Ingrese a la pestaña de contraseña o verificación 2FA y guarde los cambios.",
                ],
            )

        if "priorizacion" in pregunta_norm or "normativa" in pregunta_norm:
            return (
                "La priorización normativa evalúa cada bien con reglas deterministas basadas en la R.R. N° 449/2013 (bajas), R.R. N° 567/2018 (revalúo) y D.S. 24051.",
                [
                    "Activos con valor residual cercano a Bs 1,00 pero estado físico Bueno o Regular se sugieren para Revalúo Técnico.",
                    "Activos en estado Malo o Inutilizado con desgaste severo se clasifican como candidatos a Baja Patrimonial.",
                ],
            )

        if "baja" in pregunta_norm and not any(r.ruta == "/formularios/baja" for r in self._rutas_permitidas(payload)):
            return (
                "Su perfil de funcionario no registra bajas directas en inventario, pero sí puede emitir una Carta Formal de Solicitud de Baja hacia Activo Fijo.",
                [
                    "Abra 'Documentos y Actas' mediante el acceso directo de abajo.",
                    "Haga clic en 'Redactar Carta IA', seleccione el activo deteriorado y descargue su carta en Word o PDF para remitirla.",
                ],
            )

        if rutas_sugeridas:
            principal = rutas_sugeridas[0]
            return (
                principal.guia_breve,
                [
                    f"Use el acceso directo '{principal.titulo}' mostrado abajo.",
                    "Complete los campos o filtros requeridos en la pantalla correspondiente.",
                ],
            )

        return (
            "Puedo ayudarle a ubicar cualquier formulario del sistema, redactar cartas de solicitud/corrección, generar reportes IA o consultar normativas UAGRM.",
            [
                "Consulte por ejemplo: '¿Dónde está el formulario de baja?', '¿Cómo cambio mi contraseña?' o '¿Cómo redacto una carta de corrección?'.",
                "También puede usar el botón de micrófono para dictar su consulta por voz.",
            ],
        )
