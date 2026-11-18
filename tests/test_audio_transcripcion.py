import base64
import io
import struct
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient

from activofijo_ai.core.config import Settings, get_settings
from activofijo_ai.main import create_app
from activofijo_ai.modules.audio.engine import ServicioTranscripcionVoz
from activofijo_ai.modules.audio.router import get_servicio_transcripcion_voz

CLAVE_TEST = "clave_interna_audio_test_segura_32_caracteres!"


@dataclass
class _SegmentoFake:
    text: str


@dataclass
class _InfoFake:
    language: str = "es"


class _WhisperFake:
    def transcribe(self, audio, **_kwargs):
        assert audio.size >= 1600
        return [_SegmentoFake(text="Activos en estado malo de FICCT")], _InfoFake()


def _wav_base64(duracion_seg: float = 0.5) -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        muestras = int(16000 * duracion_seg)
        wf.writeframes(b"".join(struct.pack("<h", 0) for _ in range(muestras)))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_transcribir_audio_exige_clave_interna(tmp_path: Path) -> None:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcribir",
            json={"audio_base64": _wav_base64(), "mime_type": "audio/wav"},
        )
        assert res.status_code == 401


def test_transcribir_audio_rechaza_base64_corrupto(tmp_path: Path) -> None:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_servicio_transcripcion_voz] = lambda: ServicioTranscripcionVoz(
        modelo_inyectado=_WhisperFake()
    )

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcribir",
            headers={"X-AI-Internal-Key": CLAVE_TEST},
            json={"audio_base64": "no-es-audio-valido@@@" * 6, "mime_type": "audio/webm"},
        )
        assert res.status_code == 422


def test_transcribir_audio_wav_en_ram_devuelve_texto(tmp_path: Path) -> None:
    app = create_app()
    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_servicio_transcripcion_voz] = lambda: ServicioTranscripcionVoz(
        nombre_modelo="base",
        modelo_inyectado=_WhisperFake(),
    )

    with TestClient(app) as client:
        res = client.post(
            "/v1/audio/transcribir",
            headers={"X-AI-Internal-Key": CLAVE_TEST},
            json={
                "audio_base64": f"data:audio/wav;base64,{_wav_base64(0.6)}",
                "mime_type": "audio/wav",
                "idioma": "es-BO",
            },
        )
        assert res.status_code == 200
        body = res.json()
        assert body["texto"] == "Activos en estado malo de FICCT"
        assert body["motor_utilizado"] == "faster_whisper_local"
        assert body["modelo"] == "whisper-base-int8"
        assert body["idioma_detectado"] == "es"
