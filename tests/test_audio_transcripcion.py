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


class _WhisperInspeccionFake:
    def __init__(self, texto_fijo: str) -> None:
        self._texto_fijo = texto_fijo

    def transcribe(self, audio, **_kwargs):
        assert audio.size >= 1600
        return [_SegmentoFake(text=self._texto_fijo)], _InfoFake()


def test_interpretar_inspeccion_voz_casos_campo(tmp_path: Path) -> None:
    casos = [
        ("esta en buen esta", "dictado_inicial", None, "BUENO", "", "completar", False),
        (
            "esta en mal estado le falta una pata a la mesa",
            "dictado_inicial",
            "MESA DE MADERA",
            "MALO",
            "Le falta una pata a la mesa",
            "completar",
            False,
        ),
        (
            "le falta una pasta",
            "dictado_inicial",
            "MESA DE MADERA",
            "MALO",
            "Le falta una pata",
            "completar",
            False,
        ),
        (
            "no aprende",
            "dictado_inicial",
            "PROYECTOR EPSON",
            "MALO",
            "No enciende",
            "completar",
            False,
        ),
        ("le cambiaron el color", "dictado_inicial", None, None, "Le cambiaron el color", "completar", False),
        ("enviar", "dictado_inicial", None, None, "", "confirmar", True),
        ("aceptar", "dictado_inicial", None, None, "", "confirmar", True),
        ("confirmar y guardar", "dictado_inicial", None, None, "", "confirmar", True),
        ("ya está todo correcto por favor mándalo de una vez", "dictado_inicial", None, None, "", "confirmar", True),
        ("sí así déjalo y guárdalo", "esperando_confirmacion", None, None, "", "confirmar", True),
        ("sí, está bien, mándalo", "esperando_confirmacion", None, None, "", "confirmar", True),
        ("listo hermano envíalo nomás", "esperando_confirmacion", None, None, "", "confirmar", True),
        ("buenos días", "dictado_inicial", None, None, "", "ignorar", False),
        ("¿Hola, cómo estás?", "dictado_inicial", None, None, "", "ignorar", False),
        ("un ratito espérame ingeniero", "dictado_inicial", None, None, "", "ignorar", False),
        ("oye Juan pásame mi mochila que hace frío", "dictado_inicial", "MESA DE MADERA", None, "", "ignorar", False),
        ("¿a qué hora vamos a ir a almorzar hoy?", "dictado_inicial", "PROYECTOR EPSON", None, "", "ignorar", False),
        ("mañana tengo que comprar pan en el mercado", "dictado_inicial", "SILLA GIRATORIA", None, "", "ignorar", False),
        (
            "hola María cómo estás, tiene una esquina astillada",
            "dictado_inicial",
            "MESA DE MADERA",
            None,
            "Tiene una esquina astillada",
            "completar",
            False,
        ),
        ("buenos días, regular, raspado en esquina", "dictado_inicial", None, "REGULAR", "Raspado en esquina", "completar", False),
        ("cancelar", "dictado_inicial", None, None, "", "cancelar", False),
    ]

    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    for texto_dicho, etapa, desc_activo, est_esp, obs_esp, accion_esp, conf_esp in casos:
        app = create_app()
        app.dependency_overrides[get_settings] = lambda s=settings: s
        app.dependency_overrides[get_servicio_transcripcion_voz] = lambda t=texto_dicho: ServicioTranscripcionVoz(
            nombre_modelo="base",
            modelo_inyectado=_WhisperInspeccionFake(t),
        )
        with TestClient(app) as client:
            res = client.post(
                "/v1/audio/interpretar-inspeccion",
                headers={"X-AI-Internal-Key": CLAVE_TEST},
                json={
                    "audio_base64": _wav_base64(0.6),
                    "mime_type": "audio/wav",
                    "idioma": "es-BO",
                    "etapa_modal": etapa,
                    **({"activo_descripcion": desc_activo} if desc_activo else {}),
                },
            )
        assert res.status_code == 200
        body = res.json()
        assert body["estado_fisico"] == est_esp
        assert body["observacion"] == obs_esp
        assert body["accion"] == accion_esp
        assert body["confirma_directo"] is conf_esp
        assert body["motor_interpretacion"] == "reglas_rapidas"


def test_audio_filtra_alucinaciones_y_alta_probabilidad_no_habla() -> None:
    @dataclass
    class _SegMeta:
        text: str
        no_speech_prob: float = 0.0
        avg_logprob: float = -0.2

    class _WhisperRuidoFake:
        def transcribe(self, audio, **_kwargs):
            return [
                _SegMeta(text="[ladrido]", no_speech_prob=0.1),
                _SegMeta(text="Gracias por ver el video", no_speech_prob=0.2),
                _SegMeta(text="texto fantasma viento", no_speech_prob=0.88),
            ], _InfoFake()

    servicio = ServicioTranscripcionVoz(nombre_modelo="base", modelo_inyectado=_WhisperRuidoFake())
    import numpy as np

    resultado = servicio.transcribir(np.zeros(4000, dtype=np.float32), "es", modo_inspeccion=True)
    assert resultado.texto == ""


def test_observacion_larga_se_resume_concisamente() -> None:
    from activofijo_ai.modules.audio.inspeccion_nlp import (
        ContextoInspeccionVoz,
        interpretar_inspeccion_rapida,
    )

    res = interpretar_inspeccion_rapida(
        "eh este mira el mueble parece que está en mal estado porque tiene un golpe fuerte en el borde derecho y le falta una pata trasera y por favor mándalo",
        ContextoInspeccionVoz(activo_descripcion="ESCRITORIO DE MADERA"),
    )
    assert res.accion == "completar"
    assert res.estado_fisico == "MALO"
    assert 3 <= len(res.observacion.split()) <= 12
    assert "eh" not in res.observacion.lower().split()


async def test_respaldo_llm_inyecta_solo_skills_moviles_y_contexto(tmp_path: Path) -> None:
    import json

    import httpx

    from activofijo_ai.modules.audio.inspeccion_nlp import (
        ContextoInspeccionVoz,
        interpretar_inspeccion_con_respaldo_ia,
    )

    prompts_capturados: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        prompts_capturados.append(body["messages"][0]["content"])
        return httpx.Response(
            200,
            json={
                "model": "uagrm-activo-fijo",
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "accion": "completar",
                                    "estado_fisico": "REGULAR",
                                    "observacion": "Tornillo suelto en soporte",
                                }
                            )
                        }
                    }
                ],
                "usage": {"total_tokens": 42},
            },
        )

    settings = Settings(env="test", ai_internal_api_key=CLAVE_TEST, models_dir=tmp_path)
    res = await interpretar_inspeccion_con_respaldo_ia(
        "creo que se le salió un tornillito y baila un poco",
        settings,
        ContextoInspeccionVoz(
            etapa_modal="dictado_inicial",
            activo_codigo="U01000045",
            activo_descripcion="MESA DE MADERA",
        ),
        transport=httpx.MockTransport(handler),
    )
    assert res.motor_interpretacion == "llm_local"
    assert res.accion == "completar"
    assert res.estado_fisico == "REGULAR"
    assert res.observacion == "Tornillo suelto en soporte"
    assert len(prompts_capturados) == 1
    assert 'Responde SOLO JSON: {"a":' in prompts_capturados[0]
    assert "cartas" not in prompts_capturados[0].lower()
