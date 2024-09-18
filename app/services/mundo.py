import threading
import time
import numpy as np
from sqlalchemy.orm import Session
from app.database import get_db
from app.models import Entidad as EntidadModel
from app.services.entidad import EntidadIA 

class Mundo:
    def __init__(self):
        self.entidades = []
        self.thread = None
        self.running = False
        self.ancho = 500
        self.alto = 500
        self.recursos = self.generar_recursos()
        self.tiempo = 0
        self.temperatura = 20

    def iniciar(self):
        self.running = True
        self.poblar_mundo(10)  # Crear 10 entidades al iniciar
        self.thread = threading.Thread(target=self.bucle_principal)
        self.thread.start()

    def bucle_principal(self):
        while self.running:
            self.actualizar()
            time.sleep(1)  # Actualizar cada segundo

    def actualizar(self):
        self.tiempo += 1
        self.actualizar_temperatura()
        db = next(get_db())
        for entidad in self.entidades:
            entidad.actualizar(self)
            db_entidad = db.query(EntidadModel).filter(EntidadModel.nombre == entidad.nombre).first()
            if db_entidad:
                db_entidad.posicion_x = entidad.posicion_x
                db_entidad.posicion_y = entidad.posicion_y
                db_entidad.energia = entidad.energia
                db_entidad.puntuacion = entidad.puntuacion
        db.commit()

    def obtener_estado(self):
        return {
            'entidades': [entidad.to_dict() for entidad in self.entidades],
            'recursos': self.recursos,
            'tiempo': self.tiempo,
            'temperatura': self.temperatura,
            'estadisticas': {
                'total_entidades': len(self.entidades),
                'total_comida': len(self.recursos['comida']),
                'total_arboles': len(self.recursos['arboles']),
                'promedio_energia': sum(e.energia for e in self.entidades) / len(self.entidades) if self.entidades else 0,
                'promedio_puntuacion': sum(e.puntuacion for e in self.entidades) / len(self.entidades) if self.entidades else 0,
                }
            }

    def poblar_mundo(self, num_entidades):
        db = next(get_db())
        for i in range(num_entidades):
            x = np.random.randint(-250, 250)
            y = np.random.randint(-250, 250)
            entidad = EntidadIA(f"Entidad_{i}", x, y, 100)
            db_entidad = EntidadModel(**entidad.to_dict())
            db.add(db_entidad)
            self.entidades.append(entidad)
        db.commit()
        print(f"Se han creado {num_entidades} entidades en el mundo.")

    def generar_recursos(self):
        recursos = {'comida': [], 'agua': [], 'arboles': []}
        for _ in range(50):  # Generar 50 puntos de comida, agua y árboles
            recursos['comida'].append((np.random.randint(-250, 250), np.random.randint(-250, 250)))
            recursos['agua'].append((np.random.randint(-250, 250), np.random.randint(-250, 250)))
            recursos['arboles'].append((np.random.randint(-250, 250), np.random.randint(-250, 250)))
        return recursos

    def obtener_recursos_cercanos(self, x, y):
        comida_cercana = sum(1 for rx, ry in self.recursos['comida'] if abs(x-rx) < 50 and abs(y-ry) < 50)
        agua_cercana = sum(1 for rx, ry in self.recursos['agua'] if abs(x-rx) < 50 and abs(y-ry) < 50)
        arboles_cercanos = sum(1 for rx, ry in self.recursos['arboles'] if abs(x-rx) < 50 and abs(y-ry) < 50)
        return {'comida': comida_cercana, 'agua': agua_cercana, 'arboles': arboles_cercanos}

    def obtener_entidades_cercanas(self, x, y):
        aliados = sum(1 for e in self.entidades if abs(x-e.posicion_x) < 50 and abs(y-e.posicion_y) < 50)
        return {'aliados': aliados, 'enemigos': 0}  # Por ahora no hay enemigos

    def obtener_tiempo_del_dia(self):
        return (self.tiempo % 24) / 24.0

    def obtener_temperatura(self):
        return self.temperatura

    def obtener_peligro(self, x, y):
        # Por ahora, el peligro es aleatorio
        return np.random.random()

    def comer(self, x, y):
        for i, (rx, ry) in enumerate(self.recursos['comida']):
            if abs(x-rx) < 10 and abs(y-ry) < 10:
                del self.recursos['comida'][i]
                self.recursos['comida'].append((np.random.randint(-250, 250), np.random.randint(-250, 250)))
                return 5  # Recompensa por comer
        return 0

    def actualizar_temperatura(self):
        hora = self.tiempo % 24
        self.temperatura = 20 + 10 * np.sin(np.pi * hora / 12)  # Varía entre 10 y 30 grados