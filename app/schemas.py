from pydantic import BaseModel

class EntidadSchema(BaseModel):
    id: int
    nombre: str
    posicion_x: float
    posicion_y: float
    energia: float
    puntuacion: float

    class Config:
        orm_mode = True