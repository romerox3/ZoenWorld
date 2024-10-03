from sqlalchemy import Column, Integer, Float, String, JSON, DateTime
from app.database import Base
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone

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
    genes = Column(JSON)
    generacion = Column(Integer, default=1)
    acciones_tomadas = Column(JSON, default={})
    edad = Column(Integer, default=0)
    hambre = Column(Float, default=0)
    sed = Column(Float, default=0)
    cambio_puntuacion = Column(Float, default=0)
    cambio_energia = Column(Float, default=0)
    interacciones_recientes = Column(JSON, default={})
    logs = relationship("Log", back_populates="entidad")
    pesos_red_neuronal = Column(JSON)
    padre_id = Column(Integer, ForeignKey('entidades.id'))
    madre_id = Column(Integer, ForeignKey('entidades.id'))

class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    tiempo = Column(DateTime(timezone=True), default=datetime.now(timezone.utc))
    accion = Column(String, index=True)
    detalles = Column(String)
    entidad_id = Column(Integer, ForeignKey('entidades.id'))

    entidad = relationship("Entidad", back_populates="logs")