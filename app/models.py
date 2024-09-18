from sqlalchemy import Column, Integer, Float, String
from app.database import Base

class Entidad(Base):
    __tablename__ = "entidades"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String, index=True)
    posicion_x = Column(Float)
    posicion_y = Column(Float)
    energia = Column(Float)
    puntuacion = Column(Float, default=0)
    recompensa_promedio = Column(Float, default=0)
    perdida_promedio = Column(Float, default=0)
    epsilon = Column(Float, default=1.0)
