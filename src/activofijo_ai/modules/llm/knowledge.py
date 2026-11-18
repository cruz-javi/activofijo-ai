from dataclasses import dataclass


@dataclass(frozen=True)
class PlantillaCartaNormativa:
    referencia_base: str
    titulo_documento: str
    normas: tuple[str, ...]
    destinatario_default: str
    cargo_destinatario_default: str


PLANTILLAS_CARTAS: dict[str, PlantillaCartaNormativa] = {
    "SOLICITUD_CORRECCION_DATOS": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE CORRECCIÓN Y ACTUALIZACIÓN DE DATOS DE ACTIVO FIJO",
        titulo_documento="Solicitud de Corrección de Registro Patrimonial",
        normas=("Ley N° 1178 (SAFCO)", "D.S. N° 0181 (NB-SABS)"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Dirección Administrativa y Financiera (DUAF) — UAGRM",
    ),
    "SOLICITUD_BAJA": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE EVALUACIÓN Y BAJA PATRIMONIAL DE ACTIVO FIJO",
        titulo_documento="Solicitud de Baja Patrimonial",
        normas=("R.R. N° 449/2013 Art. 10 (Manual de Baja de Bienes)", "D.S. N° 0181"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Comisión Técnica de Bajas (DMMU / UTALAB) — UAGRM",
    ),
    "SOLICITUD_INVENTARIO": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE VERIFICACIÓN E INVENTARIO FÍSICO DE ACTIVOS FIJOS",
        titulo_documento="Solicitud de Levantamiento e Inventario Físico",
        normas=("D.S. N° 0181 (Control y Conciliación Física)", "Ley N° 1178"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Unidad de Control Patrimonial — UAGRM",
    ),
    "SOLICITUD_REASIGNACION": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE REASIGNACIÓN Y TRANSFERENCIA DE CUSTODIA DE ACTIVO FIJO",
        titulo_documento="Solicitud de Reasignación de Custodia",
        normas=("D.S. N° 0181 (Responsabilidad por la Custodia)", "Ley N° 1178"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Dirección Administrativa y Financiera (DUAF) — UAGRM",
    ),
    "SOLICITUD_REVALUO": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE REVALORIZACIÓN TÉCNICA PERICIAL DE ACTIVO FIJO",
        titulo_documento="Solicitud de Revalúo Técnico",
        normas=("R.R. N° 567/2018 V2 (UAGRM-MRVT-AF)", "NICSP 17 / NC 4", "D.S. N° 24051"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Comisión de Revalorización Técnica — UAGRM",
    ),
    "SOLICITUD_NUEVO_ACTIVO": PlantillaCartaNormativa(
        referencia_base="SOLICITUD DE DOTACIÓN E INCORPORACIÓN DE NUEVO ACTIVO FIJO",
        titulo_documento="Solicitud de Nuevo Equipamiento / Activo",
        normas=("D.S. N° 0181 (Normas Básicas del Sistema de Administración de Bienes)",),
        destinatario_default="Dirección Universitaria de Administración y Finanzas (DUAF)",
        cargo_destinatario_default="Departamento de Activos Fijos — UAGRM",
    ),
    "DESCARGO_INCIDENTE": PlantillaCartaNormativa(
        referencia_base="INFORME DE DESCARGO Y NOTIFICACIÓN DE INCIDENTE SOBRE BIEN EN CUSTODIA",
        titulo_documento="Informe de Descargo de Custodio",
        normas=("R.R. N° 449/2013 Art. 8 y Art. 12", "Ley N° 1178 (Responsabilidad Administrativa)"),
        destinatario_default="Jefatura del Departamento de Activos Fijos",
        cargo_destinatario_default="Dirección Administrativa y Financiera (DUAF) — UAGRM",
    ),
    "INFORME_TECNICO": PlantillaCartaNormativa(
        referencia_base="INFORME TÉCNICO PERICIAL DE EVALUACIÓN PATRIMONIAL",
        titulo_documento="Informe Técnico Pericial de Activo Fijo",
        normas=("R.R. N° 449/2013", "R.R. N° 567/2018 V2", "D.S. N° 0181"),
        destinatario_default="Dirección Universitaria de Administración y Finanzas (DUAF)",
        cargo_destinatario_default="Departamento de Activos Fijos — UAGRM",
    ),
}


@dataclass(frozen=True)
class RutaSistemaGuia:
    ruta: str
    titulo: str
    descripcion: str
    permiso_requerido: str | None
    palabras_clave: tuple[str, ...]
    guia_breve: str


GUIA_RUTAS_SISTEMA: tuple[RutaSistemaGuia, ...] = (
    RutaSistemaGuia(
        ruta="/dashboard",
        titulo="Panel Principal (Inicio)",
        descripcion="Resumen general y accesos rápidos según su perfil.",
        permiso_requerido=None,
        palabras_clave=("inicio", "dashboard", "principal", "menu"),
        guia_breve="Desde el Panel Principal puede acceder a los módulos habilitados para su rol.",
    ),
    RutaSistemaGuia(
        ruta="/gestion-documental",
        titulo="Documentos y Actas",
        descripcion="Imprimir actas oficiales y redactar cartas asistidas por IA (Word/PDF).",
        permiso_requerido="activos:consultar",
        palabras_clave=("acta", "actas", "imprimir", "documento", "carta", "correccion", "solicitud", "word", "pdf"),
        guia_breve="En Documentos y Actas puede consultar sus actas de asignación y usar 'Redactar Carta IA' para generar notas formales.",
    ),
    RutaSistemaGuia(
        ruta="/reportes-ia",
        titulo="Reportes IA",
        descripcion="Generación de reportes inteligentes por voz o lenguaje natural con exportación oficial PDF/Excel.",
        permiso_requerido="activos:consultar",
        palabras_clave=("reporte", "reportes", "excel", "analisis", "analitica", "padron", "estadistica", "ia", "exportar"),
        guia_breve="Ingrese a Reportes IA desde el menú lateral para consultar activos por lenguaje natural o voz y exportar el informe en PDF o Excel.",
    ),
    RutaSistemaGuia(
        ruta="/activos",
        titulo="Consulta de Activos",
        descripcion="Catálogo patrimonial, búsqueda, ficha técnica y exportación tabular.",
        permiso_requerido="activos:consultar",
        palabras_clave=("activo", "activos", "catalogo", "buscar", "ficha"),
        guia_breve="En Consultas -> Consulta de Activos puede filtrar bienes y revisar o exportar cada ficha del inventario.",
    ),
    RutaSistemaGuia(
        ruta="/trazabilidad",
        titulo="Trazabilidad de Activos",
        descripcion="Línea de tiempo e integridad criptográfica SHA-256 del ciclo de vida.",
        permiso_requerido="activos:consultar",
        palabras_clave=("trazabilidad", "historial", "hash", "sha", "auditoria", "movimientos", "cadena"),
        guia_breve="En Consultas -> Trazabilidad de Activos puede auditar todos los eventos históricos e integridad SHA-256 de un bien.",
    ),
    RutaSistemaGuia(
        ruta="/normativa",
        titulo="Motor Normativo",
        descripcion="Clasificación determinista R.R. 449/2013 y R.R. 567/2018 (Bajas y Revalúos).",
        permiso_requerido="normativa:consultar",
        palabras_clave=("normativa", "priorizacion", "regla", "reglas", "candidato", "449", "567", "24051"),
        guia_breve="El Motor Normativo evalúa vida útil, valor residual Bs 1,00 y estado físico según los reglamentos R.R. 449/2013 y R.R. 567/2018.",
    ),
    RutaSistemaGuia(
        ruta="/metricas",
        titulo="Métricas y Random Forest",
        descripcion="KPIs institucionales y proyección predictiva Random Forest.",
        permiso_requerido="metricas:consultar",
        palabras_clave=("metrica", "metricas", "random forest", "pronostico", "presupuesto", "prediccion", "estadistica"),
        guia_breve="En Métricas se muestran los KPIs patrimoniales y las proyecciones de bajas y presupuesto con Random Forest.",
    ),
    RutaSistemaGuia(
        ruta="/formularios/alta",
        titulo="Alta de Activos",
        descripcion="Registro de incorporación de nuevos activos fijos al inventario.",
        permiso_requerido="activos:crear",
        palabras_clave=("alta", "registrar", "nuevo activo", "incorporar", "incorporacion"),
        guia_breve="Ingrese a Formularios -> Alta de Activos para registrar una nueva incorporación al inventario.",
    ),
    RutaSistemaGuia(
        ruta="/formularios/asignacion",
        titulo="Asignación de Custodia",
        descripcion="Entrega formal de custodia de activos a funcionarios y oficinas.",
        permiso_requerido="activos:asignar",
        palabras_clave=("asignar", "asignacion", "custodia", "custodio", "entregar"),
        guia_breve="En Formularios -> Asignación de Custodia puede asignar uno o más bienes disponibles a un funcionario.",
    ),
    RutaSistemaGuia(
        ruta="/formularios/modificacion",
        titulo="Modificación de Activo",
        descripcion="Actualización controlada de serie, marca, modelo o descripción técnica.",
        permiso_requerido="activos:modificar",
        palabras_clave=("modificar", "modificacion", "corregir", "editar", "serie", "marca"),
        guia_breve="En Formularios -> Modificación de Activo se actualizan los datos técnicos conservando respaldo en bitácora.",
    ),
    RutaSistemaGuia(
        ruta="/formularios/baja",
        titulo="Baja de Activos",
        descripcion="Desincorporación formal de bienes según R.R. 449/2013 sin borrado físico.",
        permiso_requerido="activos:baja",
        palabras_clave=("baja", "bajas", "desincorporar", "inservible", "obsoleto", "retiro"),
        guia_breve="En Formularios -> Baja de Activos se procesa la desincorporación reglamentaria con causal y acta oficial.",
    ),
    RutaSistemaGuia(
        ruta="/formularios/revaluo",
        titulo="Revalúo Técnico",
        descripcion="Actualización de valor contable y ampliación de vida útil pericial.",
        permiso_requerido="activos:revaluo",
        palabras_clave=("revaluo", "revalorizacion", "tasacion", "vida util", "pericial"),
        guia_breve="En Formularios -> Revalúo Técnico se registran los dictámenes periciales que extienden valor y vida útil.",
    ),
    RutaSistemaGuia(
        ruta="/etiquetas",
        titulo="Identificadores y Etiquetas",
        descripcion="Impresión individual o por lotes de etiquetas QR y Código 128.",
        permiso_requerido="etiquetas:gestionar",
        palabras_clave=("etiqueta", "etiquetas", "qr", "codigo de barras", "rotulado", "termica", "reponer"),
        guia_breve="En Codificación -> Identificadores y Etiquetas puede emitir o reponer códigos QR/Code128.",
    ),
    RutaSistemaGuia(
        ruta="/inspecciones",
        titulo="Levantamientos de Campo",
        descripcion="Revisión de planillas sincronizadas desde la aplicación móvil con firma digital.",
        permiso_requerido="inspeccion:consultar",
        palabras_clave=("inspeccion", "inspecciones", "levantamiento", "campo", "movil", "cuadrilla"),
        guia_breve="En Consultas -> Levantamientos revisa y exporta las planillas físicas enviadas desde campo.",
    ),
    RutaSistemaGuia(
        ruta="/seguridad",
        titulo="Seguridad de Cuenta (Contraseña y 2FA)",
        descripcion="Cambio de contraseña y configuración de verificación en dos pasos (TOTP).",
        permiso_requerido=None,
        palabras_clave=("contraseña", "contrasena", "password", "clave", "2fa", "totp", "autenticador", "seguridad", "perfil"),
        guia_breve="Para cambiar su contraseña o activar/configurar 2FA, abra Seguridad de Cuenta en el menú de su perfil (parte inferior izquierda del menú lateral).",
    ),
    RutaSistemaGuia(
        ruta="/usuarios",
        titulo="Usuarios y Accesos",
        descripcion="Administración de cuentas institucionales y vínculo legado.",
        permiso_requerido="usuarios:gestionar",
        palabras_clave=("usuario", "usuarios", "cuenta", "cuentas", "vincular"),
        guia_breve="En Seguridad y Auditoría -> Usuarios y Accesos se crean cuentas y se vinculan códigos de funcionario.",
    ),
    RutaSistemaGuia(
        ruta="/auditoria",
        titulo="Bitácora del Sistema",
        descripcion="Registro forense de eventos e impresiones.",
        permiso_requerido="bitacora:consultar",
        palabras_clave=("bitacora", "log", "eventos", "forense"),
        guia_breve="En Seguridad y Auditoría -> Bitácora del Sistema consulta los eventos auditados con hash SHA-256.",
    ),
)

PALABRAS_DOMINIO_VALIDO: tuple[str, ...] = (
    "activo", "activos", "baja", "alta", "revaluo", "revalorizacion", "asignacion", "asignar",
    "custodia", "custodio", "transferencia", "reasignacion", "etiqueta", "qr", "codigo",
    "normativa", "priorizacion", "regla", "metricas", "random forest", "pronostico",
    "levantamiento", "inspeccion", "trazabilidad", "historial", "hash", "informe",
    "reporte", "pdf", "word", "docx", "excel", "carta", "solicitud", "correccion",
    "contraseña", "contrasena", "password", "clave", "2fa", "totp", "usuario", "rol",
    "permiso", "uagrm", "duaf", "formulario", "sistema", "hola", "ayuda", "como", "donde",
    "puedo", "que", "explica", "depreciacion", "ufv", "residual", "dmmu", "utalab",
    "proyector", "computadora", "laptop", "escritorio", "vehiculo", "mobiliario", "equipo",
    "ficct", "decanato", "biblioteca", "rectorado", "postgrado", "cuantos", "cuanto",
    "estado", "malo", "bueno", "regular", "valor", "presupuesto", "padron", "sha", "sha256",
)

PALABRAS_BLOQUEO_EXTERNO: tuple[str, ...] = (
    "receta", "cocina", "poema", "chiste", "futbol", "bitcoin", "criptomoneda",
    "programa en python", "codigo en java", "hackear", "novela", "cancion",
)
