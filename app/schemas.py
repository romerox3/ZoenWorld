from pydantic import BaseModel

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