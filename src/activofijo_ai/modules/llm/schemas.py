from typing import Any, Literal

from pydantic import BaseModel, Field

TipoDocumentoCarta = Literal[
    "SOLICITUD_CORRECCION_DATOS",
    "SOLICITUD_BAJA",
    "SOLICITUD_INVENTARIO",
    "SOLICITUD_REASIGNACION",
    "SOLICITUD_REVALUO",
    "SOLICITUD_NUEVO_ACTIVO",
    "DESCARGO_INCIDENTE",
    "INFORME_TECNICO",
]

ModoAccesoCarta = Literal["PUBLICO_HOME", "INTERNO_SISTEMA"]


class TrazaHerramientaMcp(BaseModel):
    herramienta: str
    argumentos: dict[str, Any] = Field(default_factory=dict)
    resumen_resultado: str
    duracion_ms: int = 0


class ContextoSeguridadMcpInput(BaseModel):
    usuario_id: str | None = None
    rol_usuario: str | None = "FUNCIONARIO"
    permisos_usuario: list[str] = Field(default_factory=list)
    codigo_empleado_legado: int | None = None
    acceso_amplio: bool = False
    sanitizar_publico: bool = False


class TurnoConversacionInput(BaseModel):
    rol: Literal["usuario", "asistente"]
    texto: str = Field(..., min_length=1, max_length=600)


class BorradorCartaInput(BaseModel):
    referencia: str | None = Field(default=None, max_length=250)
    destinatario_nombre: str | None = Field(default=None, max_length=120)
    destinatario_cargo: str | None = Field(default=None, max_length=120)
    cuerpo_parrafos: list[str] = Field(default_factory=list, max_length=8)
    despedida: str | None = Field(default=None, max_length=250)


class RedaccionAsistidaRequest(BaseModel):
    tipo_documento: TipoDocumentoCarta | None = None
    modo_acceso: ModoAccesoCarta = "INTERNO_SISTEMA"
    cod_activo: str | None = Field(default=None, max_length=50)
    instruccion_usuario: str = Field(..., min_length=5, max_length=1500)
    solicitante_nombre: str | None = Field(default=None, max_length=120)
    solicitante_cargo: str | None = Field(default=None, max_length=120)
    solicitante_unidad: str | None = Field(default=None, max_length=150)
    destinatario_nombre: str | None = Field(default=None, max_length=120)
    destinatario_cargo: str | None = Field(default=None, max_length=120)
    contexto_activo: dict[str, Any] | None = None
    contexto_normativo: dict[str, Any] | None = None
    contexto_seguridad: ContextoSeguridadMcpInput | None = None
    historial: list[TurnoConversacionInput] = Field(default_factory=list, max_length=8)
    borrador_actual: BorradorCartaInput | None = None


class RedaccionAsistidaResponse(BaseModel):
    motor_utilizado: Literal["ollama_personalizado", "llm_remoto", "respaldo_determinista"]
    modelo: str
    tipo_documento_detectado: TipoDocumentoCarta | None = None
    referencia: str
    lugar_fecha: str
    destinatario_nombre: str
    destinatario_cargo: str
    cuerpo_parrafos: list[str]
    despedida: str
    incluye_anexo_tecnico: bool
    fundamento_normativo: list[str] = Field(default_factory=list)
    advertencias: list[str] = Field(default_factory=list)
    trazas_mcp: list[TrazaHerramientaMcp] = Field(default_factory=list)
    tokens_usados: int = 0


class ResumenLoteReporteInput(BaseModel):
    titulo_reporte: str = Field(default="REPORTE PATRIMONIAL UAGRM", max_length=200)
    total_registros: int = Field(..., ge=0)
    valor_total_bs: float = Field(default=0.0, ge=0.0)
    estados: dict[str, int] = Field(default_factory=dict)
    grupos_top: list[dict[str, Any]] = Field(default_factory=list)
    criterios_aplicados: list[str] = Field(default_factory=list)


class ConsultaLenguajeNaturalRequest(BaseModel):
    consulta_usuario: str = Field(default="", max_length=600)
    contexto_modulo: str = Field(default="activos", max_length=60)
    consultar_padron_real: bool = False
    resumen_lote: ResumenLoteReporteInput | None = None
    contexto_seguridad: ContextoSeguridadMcpInput | None = None


class FiltrosDetectadosReporte(BaseModel):
    search: str | None = None
    estado: str | None = None
    condicion: str | None = None
    grupo_contable: str | None = None
    ubicacion: str | None = None
    valor_min: float | None = None
    valor_max: float | None = None
    gestion: int | None = None


class ConsultaLenguajeNaturalResponse(BaseModel):
    motor_utilizado: Literal["ollama_personalizado", "llm_remoto", "respaldo_determinista"]
    modelo: str
    filtros_detectados: FiltrosDetectadosReporte
    explicacion: str
    resumen_ejecutivo: str | None = None
    hallazgos_clave: list[str] = Field(default_factory=list)
    advertencias: list[str] = Field(default_factory=list)
    trazas_mcp: list[TrazaHerramientaMcp] = Field(default_factory=list)
    estadisticas_padron: dict[str, Any] | None = None
    muestras_activos: list[dict[str, Any]] = Field(default_factory=list)
    tokens_usados: int = 0


class EnlaceDirectoSistema(BaseModel):
    titulo: str
    ruta: str
    descripcion: str


class AyudaSistemaRequest(BaseModel):
    pregunta: str = Field(..., min_length=2, max_length=800)
    ruta_actual: str = Field(default="/dashboard", max_length=120)
    rol_usuario: str = Field(default="FUNCIONARIO", max_length=80)
    permisos_usuario: list[str] = Field(default_factory=list, max_length=100)
    historial: list[TurnoConversacionInput] = Field(default_factory=list, max_length=6)
    contexto_seguridad: ContextoSeguridadMcpInput | None = None


class AyudaSistemaResponse(BaseModel):
    motor_utilizado: Literal["ollama_personalizado", "llm_remoto", "respaldo_determinista"]
    modelo: str
    respuesta: str
    pasos: list[str] = Field(default_factory=list)
    enlaces_directos: list[EnlaceDirectoSistema] = Field(default_factory=list)
    fuera_de_ambito: bool = False
    trazas_mcp: list[TrazaHerramientaMcp] = Field(default_factory=list)
    filtros_navegacion: FiltrosDetectadosReporte | None = None
    tokens_usados: int = 0


class ReportePrevioInput(BaseModel):
    filtros_detectados: dict[str, Any] | None = None
    resumen_ejecutivo: str | None = Field(default=None, max_length=1200)


class ConsultaAnaliticaRequest(BaseModel):
    consulta: str = Field(..., min_length=3, max_length=800)
    modulo: str = Field(default="activos", max_length=60)
    contexto_seguridad: ContextoSeguridadMcpInput | None = None
    historial: list[TurnoConversacionInput] = Field(default_factory=list, max_length=8)
    reporte_previo: ReportePrevioInput | None = None


class ConsultaAnaliticaResponse(BaseModel):
    motor_utilizado: Literal["ollama_personalizado", "llm_remoto", "respaldo_determinista"]
    modelo: str
    respuesta: str
    filtros_detectados: FiltrosDetectadosReporte
    activos_mencionados: list[dict[str, Any]] = Field(default_factory=list)
    estadisticas: dict[str, Any] | None = None
    trazas_mcp: list[TrazaHerramientaMcp] = Field(default_factory=list)
    fuera_de_ambito: bool = False
    tokens_usados: int = 0


class DiagnosticoMetricasRequest(BaseModel):
    horizonte_meses: int = Field(default=6, ge=3, le=12)
    pregunta_usuario: str | None = Field(default=None, max_length=400)
    datos_metricas: dict[str, Any] = Field(default_factory=dict)
    contexto_seguridad: ContextoSeguridadMcpInput | None = None


class DiagnosticoMetricasResponse(BaseModel):
    motor_utilizado: Literal["ollama_personalizado", "llm_remoto", "respaldo_determinista"]
    modelo: str
    diagnostico_ejecutivo: str
    recomendaciones: list[str] = Field(default_factory=list)
    alertas_presupuestarias: list[str] = Field(default_factory=list)
    trazas_mcp: list[TrazaHerramientaMcp] = Field(default_factory=list)
    tokens_usados: int = 0
