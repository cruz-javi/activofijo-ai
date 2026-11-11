from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"]
    servicio: str
    version: str
    entorno: str


class EstadoModelo(BaseModel):
    identificador: str
    algoritmo: str
    cargado: bool
    estado: Literal["listo", "pendiente_entrenamiento"]
    ruta_esperada: str


class EstadoProveedorLlm(BaseModel):
    proveedor: str
    modelo: str
    base_url: str
    api_key_configurada: bool


class EstadoMcp(BaseModel):
    habilitado: bool
    herramientas_registradas: list[str]


class EstadoServicioIaResponse(BaseModel):
    servicio: str = "activofijo-ai"
    version: str = "0.1.0"
    estado: Literal["operativo_sin_modelos", "operativo"]
    modelos_ml: list[EstadoModelo] = Field(default_factory=list)
    llm: EstadoProveedorLlm
    mcp: EstadoMcp

