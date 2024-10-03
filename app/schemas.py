from pydantic import BaseModel
from datetime import datetime
from typing import Optional

class EntidadSchema(BaseModel):
    id: int
    nombre: str
    posicion_x: float
    posicion_y: float
    energia: float
    puntuacion: float
    recompensa_promedio: float
    perdida_promedio: float
    epsilon: float
    genes: dict
    generacion: int
    acciones_tomadas: dict
    edad: int
    hambre: float
    sed: float
    logs: list
    
    class Config:
        orm_mode = True

class LogCreate(BaseModel):
    tiempo: Optional[datetime] = None
    accion: str
    detalles: str
    entidad_id: int | None = None

    class Config:
        orm_mode = True
