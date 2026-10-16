"""Contrato de entrada del servicio: valida la solicitud antes de ejecutar nada."""
from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    pregunta: str = Field(..., min_length=3, max_length=500,
                          description="Pregunta sobre el informe de riesgo de crédito (págs. 7 a 16) o un inciso del Anexo 33.")
    usar_modelo: bool = Field(False, description="False = sin gastar crédito (ruta de abstención o «solo evidencia»). "
                                                 "True = usa el modelo de lenguaje; requiere el encabezado X-App-Token.")
