import base64
import binascii
import io
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Protocol

import av
import numpy as np
from faster_whisper import WhisperModel

FRECUENCIA_MUESTREO_WHISPER = 16_000
MAX_BYTES_AUDIO = 4_200_000
MAX_SEGUNDOS_AUDIO = 45
PROMPT_CONTEXTO_UAGRM = (
    "UAGRM, Activo Fijo, FICCT, DUAF, revalúo técnico, baja patrimonial, "
    "inventario, computadora, proyector, escritorio, código patrimonial, bolivianos."
)
PROMPT_CONTEXTO_INSPECCION_MOVIL = (
    "Inspección física UAGRM: estado bueno, regular o malo, sin novedad, rayado, roto, "
    "abollado, falta una pata, le cambiaron el color, no enciende, pantalla dañada, "
    "enviar, aceptar, confirmar, guardar, listo, cancelar."
)

_ALUCINACIONES_WHISPER_RE = re.compile(
    r"^(?:"
    r"[\s.,!?;:\-–—…\"'()\[\]*]+|"
    r"(?:\[|\()(?:m[uú]sica|music|aplausos|risas|silencio|viento|ruido|ladrido|tos|suspiro)(?:\]|\))|"
    r"gracias\s+por\s+ver(?:\s+el\s+video)?|"
    r"subt[ií]tulos\s+(?:realizados|hechos|por)\b.*|"
    r"suscr[ií]bete\b.*|"
    r"amara\.org.*|"
    r"thank\s+you(?:\s+for\s+watching)?|"
    r"you|bye|mbc\b.*"
    r")$",
    re.IGNORECASE,
)


def _es_segmento_ruido_o_alucinacion(seg: Any, texto_seg: str) -> bool:
    limpio = texto_seg.strip()
    if not limpio:
        return True
    no_speech = getattr(seg, "no_speech_prob", 0.0)
    avg_logprob = getattr(seg, "avg_logprob", 0.0)
    if isinstance(no_speech, (int, float)) and no_speech > 0.62:
        return True
    if isinstance(avg_logprob, (int, float)) and avg_logprob < -1.35:
        return True
    norm = re.sub(r"[.!?¡¿\"']+", "", limpio).strip()
    return bool(_ALUCINACIONES_WHISPER_RE.match(limpio) or (norm and _ALUCINACIONES_WHISPER_RE.match(norm)))

_MAP_FORMATOS_AV: dict[str, str] = {
    "webm": "webm",
    "ogg": "ogg",
    "opus": "ogg",
    "wav": "wav",
    "wave": "wav",
    "mp4": "mp4",
    "m4a": "mp4",
    "x-m4a": "mp4",
    "aac": "aac",
    "3gpp": "3gp",
    "3gp": "3gp",
    "mpeg": "mp3",
    "mp3": "mp3",
}


class WhisperEjecutor(Protocol):
    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> tuple[Any, Any]: ...


@dataclass(frozen=True)
class ResultadoTranscripcionVoz:
    texto: str
    idioma_detectado: str
    duracion_audio_seg: float
    modelo: str
    motor_utilizado: str
    tiempo_ms: int


def _limpiar_base64_audio(audio_base64: str) -> str:
    limpio = audio_base64.strip()
    limpio = re.sub(r"^data:audio/[a-zA-Z0-9.+;=_-]+;base64,", "", limpio, flags=re.IGNORECASE)
    return re.sub(r"\s+", "", limpio)


def _inferir_formato_av(mime_type: str) -> str | None:
    norm = (mime_type or "").lower()
    for clave, fmt in _MAP_FORMATOS_AV.items():
        if clave in norm:
            return fmt
    return None


def decodificar_audio_en_ram(audio_base64: str, mime_type: str = "audio/webm") -> np.ndarray:
    """
    Decodifica cualquier clip de audio del navegador (webm/ogg/wav/mp4) completamente en RAM
    a vector PCM float32 mono a 16 kHz listo para Whisper, sin escribir archivos temporales.
    """
    payload = _limpiar_base64_audio(audio_base64)
    if len(payload) < 64:
        raise ValueError("El audio recibido está vacío o incompleto.")

    try:
        raw_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("El contenido Base64 del audio es inválido.") from exc

    if len(raw_bytes) == 0 or len(raw_bytes) > MAX_BYTES_AUDIO:
        raise ValueError("El archivo de audio excede el tamaño máximo permitido (4 MB).")

    formato_hint = _inferir_formato_av(mime_type)
    buf = io.BytesIO(raw_bytes)

    container = None
    try:
        container = av.open(buf, mode="r")
    except av.AVError:
        if formato_hint:
            buf.seek(0)
            try:
                container = av.open(buf, mode="r", format=formato_hint)
            except av.AVError as exc:
                raise ValueError(f"No se pudo decodificar el contenedor de audio ({formato_hint}).") from exc
        else:
            raise ValueError("Formato de audio no compatible.")

    try:
        audio_streams = [s for s in container.streams if s.type == "audio"]
        if not audio_streams:
            raise ValueError("El archivo enviado no contiene pista de audio.")

        resampler = av.AudioResampler(
            format="fltp",
            layout="mono",
            rate=FRECUENCIA_MUESTREO_WHISPER,
        )
        chunks: list[np.ndarray] = []
        max_muestras = FRECUENCIA_MUESTREO_WHISPER * MAX_SEGUNDOS_AUDIO
        total_muestras = 0

        for packet in container.demux(audio_streams[0]):
            for frame in packet.decode():
                for resampled in resampler.resample(frame):
                    arr = resampled.to_ndarray().reshape(-1)
                    if arr.size > 0:
                        chunks.append(arr)
                        total_muestras += int(arr.size)
                        if total_muestras >= max_muestras:
                            break
                if total_muestras >= max_muestras:
                    break
            if total_muestras >= max_muestras:
                break

        for resampled in resampler.resample(None):
            arr = resampled.to_ndarray().reshape(-1)
            if arr.size > 0:
                chunks.append(arr)

        if not chunks:
            return np.zeros(0, dtype=np.float32)

        audio = np.concatenate(chunks).astype(np.float32)
        if audio.size > max_muestras:
            audio = audio[:max_muestras]
        return audio
    finally:
        container.close()


@lru_cache(maxsize=2)
def obtener_modelo_whisper_compartido(nombre_modelo: str, ruta_modelos: str) -> WhisperModel:
    """
    Inicializa y mantiene cacheado en RAM el modelo Whisper cuantizado (int8 CPU).
    Prioriza carga 100% local si el directorio ya fue cacheado.
    """
    whisper_dir = Path(ruta_modelos) / "whisper"
    whisper_dir.mkdir(parents=True, exist_ok=True)
    ya_cacheado = any(whisper_dir.glob(f"models--Systran--faster-whisper-{nombre_modelo}*"))
    return WhisperModel(
        nombre_modelo,
        device="cpu",
        compute_type="int8",
        cpu_threads=4,
        download_root=str(whisper_dir),
        local_files_only=ya_cacheado,
    )


class ServicioTranscripcionVoz:
    """Orquesta transcripción local de voz en español institucional UAGRM."""

    def __init__(
        self,
        nombre_modelo: str = "base",
        models_dir: Path | None = None,
        modelo_inyectado: WhisperEjecutor | None = None,
    ) -> None:
        self._nombre_modelo = nombre_modelo
        self._models_dir = models_dir or Path(__file__).resolve().parents[4] / "models"
        self._modelo_inyectado = modelo_inyectado

    def _motor(self) -> WhisperEjecutor:
        if self._modelo_inyectado is not None:
            return self._modelo_inyectado
        return obtener_modelo_whisper_compartido(self._nombre_modelo, str(self._models_dir))

    def transcribir(
        self,
        audio_pcm: np.ndarray,
        idioma: str = "es",
        modo_inspeccion: bool = False,
    ) -> ResultadoTranscripcionVoz:
        inicio = time.perf_counter()
        duracion_seg = round(float(audio_pcm.size) / FRECUENCIA_MUESTREO_WHISPER, 2)
        idioma_corto = (idioma or "es").split("-")[0].lower() or "es"

        if audio_pcm.size < 1_600:
            return ResultadoTranscripcionVoz(
                texto="",
                idioma_detectado=idioma_corto,
                duracion_audio_seg=duracion_seg,
                modelo=f"whisper-{self._nombre_modelo}-int8",
                motor_utilizado="faster_whisper_local",
                tiempo_ms=int((time.perf_counter() - inicio) * 1000),
            )

        # En ejecución real (sin modelo de prueba inyectado), omite silencio puro antes de invocar Whisper.
        if self._modelo_inyectado is None and audio_pcm.size > 0:
            rms = float(np.sqrt(np.mean(np.square(audio_pcm, dtype=np.float32))))
            if rms < 0.0025:
                return ResultadoTranscripcionVoz(
                    texto="",
                    idioma_detectado=idioma_corto,
                    duracion_audio_seg=duracion_seg,
                    modelo=f"whisper-{self._nombre_modelo}-int8",
                    motor_utilizado="faster_whisper_local",
                    tiempo_ms=int((time.perf_counter() - inicio) * 1000),
                )

        motor = self._motor()
        prompt_inicial = PROMPT_CONTEXTO_INSPECCION_MOVIL if modo_inspeccion else PROMPT_CONTEXTO_UAGRM
        usar_vad = self._modelo_inyectado is None
        kwargs_transcribe: dict[str, Any] = {
            "language": idioma_corto,
            "beam_size": 1,
            "best_of": 1,
            "temperature": 0.0,
            "vad_filter": usar_vad,
            "no_speech_threshold": 0.58,
            "log_prob_threshold": -1.1,
            "compression_ratio_threshold": 2.2,
            "condition_on_previous_text": False,
            "without_timestamps": True,
            "initial_prompt": prompt_inicial,
        }
        if usar_vad:
            kwargs_transcribe["vad_parameters"] = {
                "threshold": 0.45,
                "min_speech_duration_ms": 160,
                "min_silence_duration_ms": 300,
                "speech_pad_ms": 140,
            }

        segmentos, info = motor.transcribe(audio_pcm, **kwargs_transcribe)
        partes = [
            str(getattr(seg, "text", "") or "").strip()
            for seg in segmentos
            if not _es_segmento_ruido_o_alucinacion(seg, str(getattr(seg, "text", "") or ""))
        ]
        texto_final = " ".join(partes).strip()

        return ResultadoTranscripcionVoz(
            texto=texto_final,
            idioma_detectado=str(getattr(info, "language", idioma_corto) or idioma_corto),
            duracion_audio_seg=duracion_seg,
            modelo=f"whisper-{self._nombre_modelo}-int8",
            motor_utilizado="faster_whisper_local",
            tiempo_ms=int((time.perf_counter() - inicio) * 1000),
        )
