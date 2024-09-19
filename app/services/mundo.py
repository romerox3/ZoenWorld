import random
import time
import threading
import numpy as np
from app.database import get_db
from app.models import Entidad as EntidadModel
from app.services.entidad import EntidadIA
from app.services.configuracion import config
import string
from faker import Faker

class Mundo:
    def __init__(self):
        self.entidades = []
        self.thread = None
        self.running = False
        self.ancho = config.ANCHO_MUNDO
        self.alto = config.ALTO_MUNDO
        self.recursos = self.generar_recursos()
        self.tiempo = 0
        self.temperatura = 20
        self.tiempo_reproduccion = config.TIEMPO_REPRODUCCION
        self.logs = []

    def iniciar(self):
        self.running = True
        self.poblar_mundo(config.NUMERO_INICIAL_ENTIDADES)
        self.thread = threading.Thread(target=self.bucle_principal)
        self.thread.start()

    def bucle_principal(self):
        while self.running:
            self.actualizar()
            time.sleep(config.INTERVALO_ACTUALIZACION)

    def obtener_entidad_mas_cercana(self, x, y):
        entidades_cercanas = sorted(
            [e for e in self.entidades if e.posicion_x != x or e.posicion_y != y],
            key=lambda e: ((e.posicion_x - x)**2 + (e.posicion_y - y)**2)**0.5
        )
        return entidades_cercanas[0] if entidades_cercanas else None

    def actualizar(self):
        entidades_a_eliminar = []
        nuevas_entidades = []
        
        for entidad in self.entidades:
            entidad.actualizar(self)
            
            # Verificar si la entidad muere
            if entidad.energia <= config.MIN_ENERGIA or entidad.edad > config.EDAD_MAXIMA:
                entidades_a_eliminar.append(entidad)
            
            # Verificar si la entidad está lista para reproducirse
            elif entidad.esta_lista_para_reproducirse():
                posibles_parejas = [e for e in self.entidades if e != entidad and e.esta_lista_para_reproducirse()]
                if posibles_parejas:
                    pareja = random.choice(posibles_parejas)
                    hijo = entidad.reproducir(pareja)
                    nuevas_entidades.append(hijo)
                    entidad.energia -= config.COSTO_ENERGIA_REPRODUCCION
                    pareja.energia -= config.COSTO_ENERGIA_REPRODUCCION
        
        # Eliminar entidades muertas
        for entidad in entidades_a_eliminar:
            self.entidades.remove(entidad)
        
        # Añadir nuevas entidades
        self.entidades.extend(nuevas_entidades)
        
        self.tiempo += 1
        self.actualizar_temperatura()

    def regenerar_recursos(self):
        for tipo_recurso in ['comida', 'agua']:
            while len(self.recursos[tipo_recurso]) < config.NUMERO_INICIAL_RECURSOS:
                nuevo_id = max([r[0] for r in self.recursos[tipo_recurso]] + [0]) + 1
                self.recursos[tipo_recurso].append((nuevo_id, np.random.randint(config.MIN_X, config.MAX_X), np.random.randint(config.MIN_Y, config.MAX_Y)))

    def obtener_estado(self):
        return {
            'entidades': [entidad.to_dict() for entidad in self.entidades],
            'recursos': self.recursos,
            'tiempo': self.tiempo,
            'temperatura': self.temperatura,
            'estadisticas': {
                'total_entidades': len(self.entidades),
                'total_comida': len(self.recursos['comida']),
                'total_agua': len(self.recursos['agua']),
                'total_arboles': len(self.recursos['arboles']),
                'promedio_energia': sum(e.energia for e in self.entidades) / len(self.entidades) if self.entidades else 0,
                'promedio_puntuacion': sum(e.puntuacion for e in self.entidades) / len(self.entidades) if self.entidades else 0,
                'generacion_maxima': max(e.generacion for e in self.entidades) if self.entidades else 0,
                'mejor_puntuacion': max(e.puntuacion for e in self.entidades) if self.entidades else 0,
                'peor_puntuacion': min(e.puntuacion for e in self.entidades) if self.entidades else 0,
            },
            'mejores_entidades': sorted([entidad.to_dict() for entidad in self.entidades], key=lambda x: x['puntuacion'], reverse=True)[:5],
            'distribucion_genes': self.obtener_distribucion_genes(),
            'logs': self.logs,
            'logs_entidades': {entidad.nombre: [log.to_dict() for log in entidad.logs] for entidad in self.entidades}
        }

    def obtener_distribucion_genes(self):
        distribucion = {gen: [] for gen in self.entidades[0].genes.keys()} if self.entidades else {}
        for entidad in self.entidades:
            for gen, valor in entidad.genes.items():
                distribucion[gen].append(valor)
        return {gen: {
            'min': min(valores),
            'max': max(valores),
            'promedio': sum(valores) / len(valores)
        } for gen, valores in distribucion.items()}

    def poblar_mundo(self, num_entidades):
        db = next(get_db())
        for i in range(num_entidades):
            x = np.random.randint(config.MIN_X, config.MAX_X)
            y = np.random.randint(config.MIN_Y, config.MAX_Y)
            entidad = EntidadIA(f"{self.random_name()}", x, y, config.ENERGIA_INICIAL)
            db_entidad = EntidadModel(**entidad.to_dict())
            db.add(db_entidad)
            self.entidades.append(entidad)
        db.commit()
        print(f"Se han creado {num_entidades} entidades en el mundo.")

    def random_name(self):
        faker = Faker()
        return faker.name().split(" ")[0]

    def consumir_recurso(self, x, y, tipo_recurso, nombre_entidad=None):
        recursos_cercanos = [r for r in self.recursos[tipo_recurso] if ((r[1] - x)**2 + (r[2] - y)**2)**0.5 <= 10]
        if recursos_cercanos:
            recurso = min(recursos_cercanos, key=lambda r: ((r[1] - x)**2 + (r[2] - y)**2)**0.5)
            self.recursos[tipo_recurso] = [r for r in self.recursos[tipo_recurso] if r[0] != recurso[0]]
            if nombre_entidad:
                self.logs.append(f"{nombre_entidad} consumió {tipo_recurso}.")
            else:
                self.logs.append(f"Entidad consumió {tipo_recurso}.")
            return 1
        return 0

    def generar_recursos(self):
        recursos = {'comida': [], 'agua': [], 'arboles': []}
        for i in range(config.NUMERO_INICIAL_RECURSOS):
            recursos['comida'].append((i, np.random.randint(config.MIN_X, config.MAX_X), np.random.randint(config.MIN_Y, config.MAX_Y)))
            recursos['agua'].append((i+config.NUMERO_INICIAL_RECURSOS, np.random.randint(config.MIN_X, config.MAX_X), np.random.randint(config.MIN_Y, config.MAX_Y)))
            recursos['arboles'].append((i+2*config.NUMERO_INICIAL_RECURSOS, np.random.randint(config.MIN_X, config.MAX_X), np.random.randint(config.MIN_Y, config.MAX_Y)))
        return recursos

    def obtener_recursos_cercanos(self, x, y):
        direcciones = ['norte', 'sur', 'este', 'oeste']
        recursos = {tipo: {dir: 0 for dir in direcciones} for tipo in ['comida', 'agua', 'arboles']}
        
        for tipo in recursos:
            for _, rx, ry in self.recursos[tipo]:
                dx, dy = rx - x, ry - y
                distancia = (dx**2 + dy**2)**0.5
                if distancia < config.DISTANCIA_VISION_RECURSOS:
                    if abs(dx) > abs(dy):
                        dir = 'este' if dx > 0 else 'oeste'
                    else:
                        dir = 'norte' if dy > 0 else 'sur'
                    recursos[tipo][dir] += 1 / (distancia + 1)  # +1 para evitar división por cero
        
        return recursos

    def obtener_entidades_cercanas(self, x, y):
        direcciones = ['norte', 'sur', 'este', 'oeste']
        entidades = {dir: 0 for dir in direcciones}
        
        for entidad in self.entidades:
            if entidad.posicion_x == x and entidad.posicion_y == y:
                continue
            dx, dy = entidad.posicion_x - x, entidad.posicion_y - y
            distancia = (dx**2 + dy**2)**0.5
            if distancia < config.DISTANCIA_VISION_RECURSOS:
                if abs(dx) > abs(dy):
                    dir = 'este' if dx > 0 else 'oeste'
                else:
                    dir = 'norte' if dy > 0 else 'sur'
                entidades[dir] += 1 / (distancia + 1)  # +1 para evitar división por cero
        
        return entidades

    def obtener_tiempo_del_dia(self):
        return (self.tiempo % config.CICLO_DIA) / config.CICLO_DIA

    def obtener_temperatura(self):
        return self.temperatura

    def obtener_peligro(self, x, y):
        # Implementar lógica de peligro basada en la posición y otros factores
        return random.random()  # Por ahora, retornamos un valor aleatorio entre 0 y 1

    def actualizar_temperatura(self):
        # Simular cambios de temperatura basados en el tiempo del día
        hora_del_dia = self.obtener_tiempo_del_dia() * 24
        self.temperatura = 20 + 10 * np.sin((hora_del_dia - 6) * np.pi / 12)
        self.temperatura = max(config.MIN_TEMPERATURA, min(config.MAX_TEMPERATURA, self.temperatura))