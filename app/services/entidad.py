import random
from datetime import datetime, timezone
from collections import deque
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from app.services.configuracion import config
from faker import Faker
from app.schemas import LogCreate
from app.models import Entidad as EntidadModel
from app.database import AsyncSessionLocal
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select


# Configurar TensorFlow para usar GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU está configurada y lista para usar")
    except RuntimeError as e:
        print(e)
else:
    print("No se detectó GPU. Se usará CPU.")

@dataclass
class LogEntidad:
    tiempo: int
    accion: str
    detalles: str

    def to_dict(self):
        return {
            "tiempo": self.tiempo,
            "accion": self.accion,
            "detalles": self.detalles
        }


@dataclass
class Estado:
    # Información de la entidad
    posicion_x: float
    posicion_y: float
    energia: float
    hambre: float
    sed: float
    edad: float
    generacion: float
    cambio_puntuacion: float
    cambio_energia: float

    # Información del entorno
    comida_norte: float
    comida_sur: float
    comida_este: float
    comida_oeste: float
    agua_norte: float
    agua_sur: float
    agua_este: float
    agua_oeste: float
    arboles_norte: float
    arboles_sur: float
    arboles_este: float
    arboles_oeste: float
    aliados_norte: float
    aliados_sur: float
    aliados_este: float
    aliados_oeste: float
    tiempo_del_dia: float
    temperatura: float
    peligro: float

class EntidadIA:
    def __init__(self, nombre, posicion_x, posicion_y, energia, log_service, genes=None, id=None, puntuacion=0):
        self.id = id
        self.nombre = nombre
        self.posicion_x = posicion_x
        self.posicion_y = posicion_y
        self.energia = energia
        self.genes = genes if genes else self.generar_genes()
        self.red_neuronal = self.crear_red_neuronal()
        self.red_neuronal_objetivo = self.crear_red_neuronal()
        self.actualizar_red_objetivo()
        self.memoria = deque(maxlen=5)
        self.puntuacion = puntuacion
        self.buffer_experiencia = []
        self.epsilon = config.EPSILON_INICIAL
        self.epsilon_min = config.EPSILON_MINIMO
        self.epsilon_decay = config.EPSILON_DECAY
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.generacion = 1
        self.hambre = 0
        self.sed = 0
        self.edad = 0
        self.acciones_tomadas = {i: 0 for i in range(9)}
        self.cambio_puntuacion = 0
        self.cambio_energia = 0
        self.historial_puntuacion = deque(maxlen=10)
        self.historial_energia = deque(maxlen=10)
        self.interacciones_recientes = []
        self.logs = []
        self.padre_id = None
        self.madre_id = None
        self.log_service = log_service


    def generar_genes(self):
        return {
            'velocidad': random.uniform(config.MIN_VELOCIDAD, config.MAX_VELOCIDAD),
            'vision': random.uniform(config.MIN_VISION, config.MAX_VISION),
            'metabolismo': random.uniform(config.MIN_METABOLISMO, config.MAX_METABOLISMO),
            'agresividad': random.uniform(config.MIN_AGRESIVIDAD, config.MAX_AGRESIVIDAD),
            'sociabilidad': random.uniform(config.MIN_SOCIABILIDAD, config.MAX_SOCIABILIDAD),
            'inteligencia': random.uniform(config.MIN_INTELIGENCIA, config.MAX_INTELIGENCIA),
            'resistencia': random.uniform(config.MIN_RESISTENCIA, config.MAX_RESISTENCIA),
            'adaptabilidad': random.uniform(config.MIN_ADAPTABILIDAD, config.MAX_ADAPTABILIDAD)
        }

    def crear_red_neuronal(self):
        with tf.device('/GPU:0'):
            modelo = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(28,)),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation='relu'),
                tf.keras.layers.Dense(9, activation='softmax')
            ])
            modelo.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        return modelo

    def actualizar_red_objetivo(self):
        self.red_neuronal_objetivo.set_weights(self.red_neuronal.get_weights())

    def obtener_estado(self, mundo):
        recursos_cercanos = mundo.obtener_recursos_cercanos(self.posicion_x, self.posicion_y)
        entidades_cercanas = mundo.obtener_entidades_cercanas(self.posicion_x, self.posicion_y)
        tiempo_del_dia = mundo.obtener_tiempo_del_dia()
        temperatura = mundo.obtener_temperatura()
        peligro = mundo.obtener_peligro(self.posicion_x, self.posicion_y)
        
        def normalizar(valor, min_valor, max_valor):
            return (valor - min_valor) / (max_valor - min_valor)
        
        estado = [
            normalizar(self.posicion_x, config.MIN_X, config.MAX_X),
            normalizar(self.posicion_y, config.MIN_Y, config.MAX_Y),
            normalizar(self.energia, config.MIN_ENERGIA, config.MAX_ENERGIA),
            normalizar(self.hambre, config.MIN_HAMBRE, config.MAX_HAMBRE),
            normalizar(self.sed, config.MIN_SED, config.MAX_SED),
            normalizar(self.edad, config.MIN_EDAD, config.MAX_EDAD),
            normalizar(self.generacion, config.MIN_GENERACION, config.MAX_GENERACION),
            normalizar(self.cambio_puntuacion, config.MIN_CAMBIO_PUNTUACION, config.MAX_CAMBIO_PUNTUACION),
            normalizar(self.cambio_energia, config.MIN_CAMBIO_ENERGIA, config.MAX_CAMBIO_ENERGIA)
        ]
        
        for tipo_recurso in ['comida', 'agua', 'arboles']:
            for direccion in ['norte', 'sur', 'este', 'oeste']:
                estado.append(normalizar(recursos_cercanos[tipo_recurso][direccion], config.MIN_RECURSOS, config.MAX_RECURSOS))
        
        for direccion in ['norte', 'sur', 'este', 'oeste']:
            estado.append(normalizar(entidades_cercanas[direccion], config.MIN_RECURSOS, config.MAX_RECURSOS))
        
        estado.extend([
            normalizar(tiempo_del_dia, config.MIN_TIEMPO_DEL_DIA, config.MAX_TIEMPO_DEL_DIA),
            normalizar(temperatura, config.MIN_TEMPERATURA, config.MAX_TEMPERATURA),
            normalizar(peligro, config.MIN_PELIGRO, config.MAX_PELIGRO)
        ])
        
        return np.array(estado, dtype=np.float32)

    async def actualizar(self, mundo):
        try:
            await self.envejecer()
            self.actualizar_estado_interno()
            estado = self.obtener_estado(mundo)
            decision = self.tomar_decision(estado)
            recompensa = await self.ejecutar_accion(decision, mundo)
            self.registrar_experiencia(estado, decision, recompensa)
            await self.entrenar()
        except Exception as e:
            print(f"Error al actualizar entidad {self.id}: {str(e)}")

    def actualizar_estado_interno(self):
        self.hambre += 1
        self.sed += 1
        factor_edad = 1 + (self.edad / config.EDAD_MAXIMA) * 0.5
        consumo_energia = 0.5 * self.genes['metabolismo'] * factor_edad
        self.energia = max(config.MIN_ENERGIA, self.energia - consumo_energia)

    def debe_ser_eliminada(self):
        return (self.energia <= config.MIN_ENERGIA or 
                self.edad > config.EDAD_MAXIMA or 
                self.hambre > config.MAX_HAMBRE or 
                self.sed > config.MAX_SED)

    async def entrenar(self):
        if len(self.memoria) >= config.BATCH_SIZE:
            batch = random.sample(self.memoria, config.BATCH_SIZE)
            estados, acciones, recompensas, siguientes_estados = zip(*batch)
            self.red_neuronal.fit(np.array(estados), np.array(acciones), sample_weight=np.array(recompensas), epochs=1, verbose=0)

    async def actualizar_en_db(self):
        async with AsyncSessionLocal() as db:
            async with db.begin():
                try:
                    stmt = select(EntidadModel).where(EntidadModel.id == self.id)
                    result = await db.execute(stmt)
                    db_entidad = result.scalar_one_or_none()
                    if db_entidad:
                        for key, value in self.to_dict().items():
                            setattr(db_entidad, key, value)
                    else:
                        db_entidad = EntidadModel(**self.to_dict())
                        db.add(db_entidad)
                    await db.flush()
                except Exception as e:
                    await self.log_service.crear_log(LogCreate(
                        accion="Error",
                        detalles=f"Error al actualizar en DB: {str(e)}",
                        entidad_id=self.id
                    ))
                    raise

    @tf.function(experimental_relax_shapes=True)
    def tomar_decision(self, estado):
        estado_tensor = tf.convert_to_tensor(estado, dtype=tf.float32)
        estado_tensor = tf.cast(estado_tensor, tf.float32)
        estado_tensor = tf.expand_dims(estado_tensor, 0)
        accion = self.red_neuronal(estado_tensor)
        return tf.keras.backend.get_value(tf.squeeze(accion))

    def entrenar(self):
        if len(self.buffer_experiencia) < config.TAMANO_MINIBATCH:
            return 0

        minibatch = random.sample(self.buffer_experiencia, config.TAMANO_MINIBATCH)
        estados = np.array([exp[0] for exp in minibatch])
        acciones = np.array([exp[1] for exp in minibatch])
        recompensas = np.array([exp[2] for exp in minibatch])
        nuevos_estados = np.array([exp[3] for exp in minibatch])

        q_valores_actuales = self.red_neuronal.predict(estados)
        q_valores_futuros = self.red_neuronal_objetivo.predict(nuevos_estados)

        for i in range(config.TAMANO_MINIBATCH):
            accion = acciones[i]
            if recompensas[i] == config.PENALIZACION_ENERGIA_BAJA:  # Si la entidad murió
                q_valores_actuales[i][accion] = recompensas[i]
            else:
                q_valores_actuales[i][accion] = recompensas[i] + config.FACTOR_DESCUENTO * np.max(q_valores_futuros[i])

        history = self.red_neuronal.fit(estados, q_valores_actuales, epochs=1, verbose=0)
        loss = history.history['loss'][0]

        if self.epsilon > config.EPSILON_MINIMO:
            self.epsilon *= config.EPSILON_DECAY

        if self.edad % config.ACTUALIZACION_RED_OBJETIVO == 0:
            self.actualizar_red_objetivo()

        self.historial_perdidas.append(loss)
        if len(self.historial_perdidas) > 100:
            self.historial_perdidas.pop(0)

        return loss

    async def ejecutar_accion(self, decision, mundo):
        recompensa = 0
        consumo_energia = 1 * self.genes['metabolismo']
        
        if decision < 4:  # Movimiento
            if decision == 0:  # Mover arriba
                self.posicion_y = min(config.MAX_Y, self.posicion_y + config.DISTANCIA_MOVIMIENTO * self.genes['velocidad'])
                await self.log_service.crear_log(LogCreate(
                    accion="Mover",
                    detalles=f"Arriba: {self.posicion_y}",
                    entidad_id=self.id
                ))
            elif decision == 1:  # Mover abajo
                self.posicion_y = max(config.MIN_Y, self.posicion_y - config.DISTANCIA_MOVIMIENTO * self.genes['velocidad'])
                await self.log_service.crear_log(LogCreate(
                    accion="Mover",
                    detalles=f"Abajo: {self.posicion_y}",
                    entidad_id=self.id
                ))
            elif decision == 2:  # Mover izquierda
                self.posicion_x = max(config.MIN_X, self.posicion_x - config.DISTANCIA_MOVIMIENTO * self.genes['velocidad'])
                await self.log_service.crear_log(LogCreate(
                    accion="Mover",
                    detalles=f"Izquierda: {self.posicion_x}",
                    entidad_id=self.id
                ))
            elif decision == 3:  # Mover derecha
                self.posicion_x = min(config.MAX_X, self.posicion_x + config.DISTANCIA_MOVIMIENTO * self.genes['velocidad'])
                await self.log_service.crear_log(LogCreate(
                    accion="Mover",
                    detalles=f"Derecha: {self.posicion_x}",
                    entidad_id=self.id
                ))
            consumo_energia *= config.MULTIPLICADOR_CONSUMO_MOVIMIENTO
        elif decision == 4:  # Comer
            comida_consumida = mundo.consumir_recurso(x=self.posicion_x, y=self.posicion_y, tipo_recurso='comida', nombre_entidad=self.nombre)
            self.energia = min(config.MAX_ENERGIA, self.energia + comida_consumida * config.ENERGIA_POR_COMIDA)
            self.hambre = max(config.MIN_HAMBRE, self.hambre - comida_consumida * config.REDUCCION_HAMBRE_POR_COMIDA)
            recompensa += comida_consumida * config.RECOMPENSA_COMIDA
            await self.log_service.crear_log(LogCreate(
                accion="Comer",
                detalles=f"Comida consumida: {comida_consumida}, Energía: {self.energia}, Hambre: {self.hambre}, Recompensa: {recompensa}",
                entidad_id=self.id
            ))
        elif decision == 5:  # Beber
            agua_consumida = mundo.consumir_recurso(x=self.posicion_x, y=self.posicion_y, tipo_recurso='agua', nombre_entidad=self.nombre)
            self.energia = min(config.MAX_ENERGIA, self.energia + agua_consumida * config.ENERGIA_POR_AGUA)
            self.sed = max(config.MIN_SED, self.sed - agua_consumida * config.REDUCCION_SED_POR_AGUA)
            recompensa += agua_consumida * config.RECOMPENSA_AGUA
            await self.log_service.crear_log(LogCreate(
                
                accion="Beber",
                detalles=f"Agua consumida: {agua_consumida}, Energía: {self.energia}, Sed: {self.sed}, Recompensa: {recompensa}",
                entidad_id=self.id
            ))
        elif decision == 6:  # Socializar
            entidad_cercana = await mundo.obtener_entidad_mas_cercana(self.posicion_x, self.posicion_y)
            if entidad_cercana:
                self.energia = max(config.MIN_ENERGIA, self.energia - config.ENERGIA_SOCIALIZACION)
                recompensa += config.RECOMPENSA_SOCIALIZACION
                await self.log_service.crear_log(LogCreate(
                    accion="Socializar",
                    detalles=f"Socialización con {entidad_cercana.nombre}",
                    entidad_id=self.id
                ))
            else:
                recompensa = config.PENALIZACION_ACCION_FALLIDA
                await self.log_service.crear_log(LogCreate(
                    accion="Socializar",
                    detalles="Intento de socialización fallido",
                    entidad_id=self.id
                ))
        elif decision == 7:  # Atacar
            entidad_cercana = await mundo.obtener_entidad_mas_cercana(self.posicion_x, self.posicion_y)
            if entidad_cercana:
                self.energia = max(config.MIN_ENERGIA, self.energia - config.ENERGIA_ATAQUE)
                entidad_cercana.energia = max(config.MIN_ENERGIA, entidad_cercana.energia - config.ENERGIA_ATAQUE)
                recompensa += config.RECOMPENSA_ATAQUE
                await self.log_service.crear_log(LogCreate(
                    accion="Atacar",
                    detalles=f"Ataque a {entidad_cercana.nombre}",
                    entidad_id=self.id
                ))
            else:
                recompensa = config.PENALIZACION_ACCION_FALLIDA
                await self.log_service.crear_log(LogCreate(
                    accion="Atacar",
                    detalles="Intento de ataque fallido",
                    entidad_id=self.id
                ))
        elif decision == 8:  # Reproducción
            entidad_cercana = await mundo.obtener_entidad_mas_cercana(self.posicion_x, self.posicion_y)
            if entidad_cercana and self.esta_lista_para_reproducirse() and entidad_cercana.esta_lista_para_reproducirse():
                hijo = self.reproducir(entidad_cercana)
                mundo.entidades.append(hijo)
                self.energia -= config.COSTO_ENERGIA_REPRODUCCION
                entidad_cercana.energia -= config.COSTO_ENERGIA_REPRODUCCION
                recompensa = config.RECOMPENSA_REPRODUCCION
                await self.log_service.crear_log(LogCreate(
                    accion="Reproducir",
                    detalles=f"Reproducción exitosa con {entidad_cercana.nombre}",
                    entidad_id=self.id
                ))
            else:
                recompensa = config.PENALIZACION_ACCION_FALLIDA
                await self.log_service.crear_log(LogCreate(
                    accion="Reproducir",
                    detalles="Intento de reproducción fallido",
                    entidad_id=self.id
                ))
        else:
            recompensa = config.PENALIZACION_ACCION_FALLIDA

        self.energia = max(config.MIN_ENERGIA, self.energia - consumo_energia)
        self.puntuacion += recompensa  # Asegúrate de actualizar la puntuación aquí
        await self.log_service.crear_log(LogCreate(
            accion="Actualizar",
            detalles=f"Energía: {self.energia}, Consumo de energía: {consumo_energia}, Puntuación: {self.puntuacion}",
            entidad_id=self.id
        ))
        self.acciones_tomadas[decision] += 1
        await self.log_service.crear_log(LogCreate(
            accion="Actualizar",
            detalles=f"Acciones tomadas: {self.acciones_tomadas}",
            entidad_id=self.id
        ))

        if self.energia <= config.MIN_ENERGIA:
            recompensa = config.PENALIZACION_ENERGIA_BAJA
            await self.log_service.crear_log(LogCreate(
                accion="Actualizar",
                detalles=f"Recompensa: {recompensa}, Energía: {self.energia}",
                entidad_id=self.id
            ))
        elif self.energia > config.UMBRAL_ENERGIA_ALTA:
            recompensa += config.RECOMPENSA_ENERGIA_ALTA
            await self.log_service.crear_log(LogCreate(
                accion="Actualizar",
                detalles=f"Recompensa: {recompensa}, Energía: {self.energia}",
                entidad_id=self.id
            ))

        return recompensa

    def ajustar_pesos(self, pesos, forma_objetivo):
        pesos_ajustados = []
        for peso, forma in zip(pesos, forma_objetivo):
            if peso.shape != forma:
                if len(peso.shape) == 2:
                    peso_ajustado = np.zeros(forma)
                    min_filas = min(peso.shape[0], forma[0])
                    min_columnas = min(peso.shape[1], forma[1])
                    peso_ajustado[:min_filas, :min_columnas] = peso[:min_filas, :min_columnas]
                else:
                    peso_ajustado = np.zeros(forma)
                    min_dim = min(peso.shape[0], forma[0])
                    peso_ajustado[:min_dim] = peso[:min_dim]
            else:
                peso_ajustado = peso
            pesos_ajustados.append(peso_ajustado)
        return pesos_ajustados

    def combinar_pesos(self, p_padre, p_madre):
        if isinstance(p_padre, list) and isinstance(p_madre, list):
            return [self.combinar_pesos(pp, pm) for pp, pm in zip(p_padre, p_madre)]
        else:
            return (np.array(p_padre) + np.array(p_madre)) / 2

    def random_name(self):
        faker = Faker()
        return faker.name().split(" ")[0]

    def reproducir(self, pareja):
        nuevos_genes = {}
        for gen in self.genes:
            if random.random() < 0.5:
                nuevos_genes[gen] = self.genes[gen]
            else:
                nuevos_genes[gen] = pareja.genes[gen]
            # Mutación
            if random.random() < config.PROBABILIDAD_MUTACION:
                nuevos_genes[gen] *= random.uniform(config.RANGO_MUTACION_MIN, config.RANGO_MUTACION_MAX)
        
        nueva_entidad = EntidadIA(f"{self.random_name()}", 
                                (self.posicion_x + pareja.posicion_x) / 2,
                                (self.posicion_y + pareja.posicion_y) / 2,
                                config.ENERGIA_INICIAL,
                                self.log_service,
                                id=None,
                                genes=nuevos_genes)
        nueva_entidad.generacion = max(self.generacion, pareja.generacion) + 1
        nueva_entidad.padre_id = self.id
        nueva_entidad.madre_id = pareja.id
        
        # Transferencia de conocimiento
        pesos_padre = self.obtener_pesos_red_neuronal()
        pesos_madre = pareja.obtener_pesos_red_neuronal()
        
        # Combinar los pesos de los padres
        pesos_hijo = []
        for p_padre, p_madre in zip(pesos_padre, pesos_madre):
            p_hijo = np.array(self.combinar_pesos(p_padre, p_madre))
            pesos_hijo.append(p_hijo)
        
        # Establecer los pesos combinados en la red neuronal del hijo
        nueva_entidad.red_neuronal.set_weights(pesos_hijo)

        return nueva_entidad

    def obtener_metricas_aprendizaje(self):
        return {
            "recompensa_promedio": np.mean(self.historial_recompensas) if self.historial_recompensas else 0,
            "perdida_promedio": np.mean(self.historial_perdidas) if self.historial_perdidas else 0,
            "epsilon": self.epsilon
        }

    def obtener_pesos_red_neuronal(self):
        return [peso.tolist() for peso in self.red_neuronal.get_weights()]

    def to_dict(self):
        metricas = self.obtener_metricas_aprendizaje()
        return {
            "nombre": self.nombre,
            "posicion_x": float(self.posicion_x),
            "posicion_y": float(self.posicion_y),
            "energia": float(self.energia),
            "puntuacion": float(self.puntuacion),
            "recompensa_promedio": float(metricas["recompensa_promedio"]),
            "perdida_promedio": float(metricas["perdida_promedio"]),
            "epsilon": float(metricas["epsilon"]),
            "genes": {k: float(v) for k, v in self.genes.items()},
            "generacion": int(self.generacion),
            "acciones_tomadas": {int(k): int(v) for k, v in self.acciones_tomadas.items()},
            "edad": int(self.edad),
            "hambre": float(self.hambre),
            "sed": float(self.sed),
            "cambio_puntuacion": float(self.cambio_puntuacion),
            "cambio_energia": float(self.cambio_energia),
            "interacciones_recientes": self.interacciones_recientes,
            "logs": [log.to_dict() for log in self.logs],
            "pesos_red_neuronal": [peso.tolist() for peso in self.red_neuronal.get_weights()],
            "padre_id": self.padre_id,
            "madre_id": self.madre_id
        }

    def esta_lista_para_reproducirse(self):
        return self.edad >= config.EDAD_REPRODUCCION and self.energia > config.ENERGIA_REPRODUCCION and self.puntuacion > config.PUNTUACION_REPRODUCCION

    async def envejecer(self):
        self.edad += 1
        energia_maxima = config.MAX_ENERGIA * (1 - self.edad / config.EDAD_MAXIMA)
        self.energia = min(self.energia, energia_maxima)
        if self.id:
            await self.log_service.crear_log(LogCreate(
                accion="Envejecer",
                detalles=f"Edad: {self.edad}, Energía máxima: {energia_maxima:.2f}",
                entidad_id=self.id
            ))
