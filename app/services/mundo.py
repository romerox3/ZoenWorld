import random
import asyncio
import time
import threading
import numpy as np
from app.database import get_db
from app.models import Entidad as EntidadModel
from app.services.entidad import EntidadIA
from app.services.configuracion import config
import string
from faker import Faker
from app.services.logs import LogService
from app.schemas import LogCreate
from contextlib import asynccontextmanager
from sqlalchemy.orm import Session
from app.database import AsyncSessionLocal
from app.schemas import LogCreate
import json
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import engine
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from datetime import datetime, timezone

class Mundo:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.entidades = []
        self.running = False
        self.alto = config.ALTO_MUNDO
        self.recursos = self.generar_recursos()
        self.tiempo = 0
        self.temperatura = 20
        self.tiempo_reproduccion = config.TIEMPO_REPRODUCCION
        self.logs = []
        self.log_service = LogService()
        self.entidades_a_eliminar = []
        self.nuevas_entidades = []

    async def iniciar(self):
        self.running = True
        await self.poblar_mundo(config.NUMERO_INICIAL_ENTIDADES)
        asyncio.create_task(self.bucle_principal())

    async def pausar(self):
        self.running = False

    async def reanudar(self):
        if not self.running:
            self.running = True
            asyncio.create_task(self.bucle_principal())

    async def reiniciar(self):
        self.entidades = []
        self.tiempo = 0
        await self.poblar_mundo(config.NUMERO_INICIAL_ENTIDADES)
        await self.log_service.crear_log(LogCreate(
            accion="Reiniciar",
            detalles="Simulación reiniciada"
        ))

    async def bucle_principal(self):
        while self.running:
            try:
                await self.actualizar()
                await asyncio.sleep(config.INTERVALO_ACTUALIZACION)
            except Exception as e:
                print(f"Error en bucle_principal: {str(e)}")
                await asyncio.sleep(1)

    async def obtener_entidad_mas_cercana(self, x, y):
        entidades_cercanas = sorted(
            [e for e in self.entidades if e.posicion_x != x or e.posicion_y != y],
            key=lambda e: ((e.posicion_x - x)**2 + (e.posicion_y - y)**2)**0.5
        )
        return entidades_cercanas[0] if entidades_cercanas else None

    async def actualizar(self):
        async with AsyncSessionLocal() as db:
            async with db.begin():
                try:
                    await asyncio.gather(*[self.actualizar_entidad(entidad, db) for entidad in self.entidades])
                    await self.procesar_cambios(db)
                except Exception as e:
                    print(f"Error en actualizar: {str(e)}")
                    await db.rollback()
                else:
                    await db.commit()

    async def actualizar_entidad(self, entidad, db):
        try:
            await entidad.actualizar(self)
            if entidad.debe_ser_eliminada():
                self.entidades_a_eliminar.append(entidad)
                await self.log_service.crear_log(LogCreate(
                    accion="Eliminar",
                    detalles=f"Entidad {entidad.nombre} marcada para eliminación",
                    entidad_id=entidad.id
                ), db)
            elif entidad.esta_lista_para_reproducirse():
                await self.intentar_reproducir(entidad, db)
        except Exception as e:
            print(f"Error al actualizar entidad {entidad.id}: {str(e)}")

    async def intentar_reproducir(self, entidad, db):
        posibles_parejas = [e for e in self.entidades if e != entidad and e.esta_lista_para_reproducirse()]
        if posibles_parejas:
            pareja = random.choice(posibles_parejas)
            hijo = entidad.reproducir(pareja)
            self.nuevas_entidades.append(hijo)
            await self.log_service.crear_log(LogCreate(
                accion="Reproducir",
                detalles=f"Entidad {entidad.nombre} se reprodujo con {pareja.nombre}",
                entidad_id=entidad.id
            ), db)

    async def procesar_cambios(self, db):
        for entidad in self.entidades_a_eliminar:
            self.entidades.remove(entidad)
        self.entidades_a_eliminar.clear()

        for entidad in self.nuevas_entidades:
            self.entidades.append(entidad)
            db_entidad = EntidadModel(**entidad.to_dict())
            db.add(db_entidad)
        self.nuevas_entidades.clear()

        for entidad in self.entidades:
            db_entidad = await db.get(EntidadModel, entidad.id)
            if db_entidad:
                for key, value in entidad.to_dict().items():
                    setattr(db_entidad, key, value)

    async def poblar_mundo(self, num_entidades):
        async with AsyncSessionLocal() as db:
            async with db.begin():
                mejores_entidades = await db.execute(select(EntidadModel).order_by(EntidadModel.puntuacion.desc()).limit(5))
                mejores_entidades = mejores_entidades.scalars().all()

                for i in range(num_entidades):
                    if i < len(mejores_entidades):
                        entidad_data = mejores_entidades[i]
                        nuevo_nombre = f"{entidad_data.nombre}_clon_{self.random_name()}"
                        entidad = EntidadIA(
                            nombre=nuevo_nombre,
                            posicion_x=entidad_data.posicion_x,
                            posicion_y=entidad_data.posicion_y,
                            energia=100,
                            genes=entidad_data.genes,
                            puntuacion=0,
                            log_service=self.log_service,
                            id=None
                        )
                        entidad.red_neuronal.set_weights([np.array(peso) for peso in entidad_data.pesos_red_neuronal])
                    else:
                        x = np.random.randint(config.MIN_X, config.MAX_X)
                        y = np.random.randint(config.MIN_Y, config.MAX_Y)
                        entidad = EntidadIA(self.random_name(), x, y, config.ENERGIA_INICIAL, self.log_service, id=None)
                    
                    db_entidad = EntidadModel(**{k: v for k, v in entidad.to_dict().items() if k != 'id'})
                    db.add(db_entidad)
                    await db.flush()
                    entidad.id = db_entidad.id
                    self.entidades.append(entidad)

    def random_name(self):
        faker = Faker()
        nombre = faker.first_name()
        apellido = faker.last_name()
        identificador = random.randint(1000, 9999)
        return f"{nombre} {apellido}"

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
        
        # Generar agua
        for i in range(config.NUMERO_INICIAL_RECURSOS):
            recursos['agua'].append((i, np.random.randint(config.MIN_X, config.MAX_X), np.random.randint(config.MIN_Y, config.MAX_Y)))
        
        # Generar árboles donde haya acumulación de agua
        for i in range(config.NUMERO_INICIAL_ARBOLES):
            agua_cercana = random.choice(recursos['agua'])
            x_arbol = agua_cercana[1] + np.random.randint(-config.DISTANCIA_ARBOLES_AGUA, config.DISTANCIA_ARBOLES_AGUA)
            y_arbol = agua_cercana[2] + np.random.randint(-config.DISTANCIA_ARBOLES_AGUA, config.DISTANCIA_ARBOLES_AGUA)
            recursos['arboles'].append((i, x_arbol, y_arbol))
        
        # Generar comida alrededor de los árboles
        for i in range(config.NUMERO_INICIAL_RECURSOS):
            arbol_cercano = random.choice(recursos['arboles'])
            x_comida = arbol_cercano[1] + np.random.randint(-config.DISTANCIA_COMIDA_ARBOLES, config.DISTANCIA_COMIDA_ARBOLES)
            y_comida = arbol_cercano[2] + np.random.randint(-config.DISTANCIA_COMIDA_ARBOLES, config.DISTANCIA_COMIDA_ARBOLES)
            recursos['comida'].append((i, x_comida, y_comida))
        
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

    def guardar_mejores_entidades(self):
        mejores_entidades = sorted(self.entidades, key=lambda e: e.puntuacion, reverse=True)[:5]
        with open("mejores_entidades.json", "w") as file:
            json.dump([entidad.to_dict() for entidad in mejores_entidades], file)

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