import random
from collections import deque
import numpy as np
import tensorflow as tf
from dataclasses import dataclass
from app.services.configuracion import config

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
    def __init__(self, nombre, posicion_x, posicion_y, energia, genes=None):
        self.nombre = nombre
        self.posicion_x = posicion_x
        self.posicion_y = posicion_y
        self.energia = energia
        self.genes = genes if genes else self.generar_genes()
        self.red_neuronal = self.crear_red_neuronal()
        self.memoria = deque(maxlen=5)
        self.puntuacion = 0
        self.buffer_experiencia = deque(maxlen=config.TAMANO_BUFFER_EXPERIENCIA)
        self.epsilon = config.EPSILON_INICIAL
        self.epsilon_min = config.EPSILON_MINIMO
        self.epsilon_decay = config.EPSILON_DECAY
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.generacion = 1
        self.hambre = 0
        self.sed = 0
        self.edad = 0
        self.acciones_tomadas = {i: 0 for i in range(8)}
        self.cambio_puntuacion = 0
        self.cambio_energia = 0
        self.historial_puntuacion = deque(maxlen=10)
        self.historial_energia = deque(maxlen=10)
        self.interacciones_recientes = []
        self.logs = []

    def agregar_log(self, accion, detalles):
        self.logs.append(LogEntidad(tiempo=self.edad, accion=accion, detalles=detalles))
        if len(self.logs) > config.MAX_LOGS_POR_ENTIDAD:
            self.logs.pop(0)

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
        inteligencia = max(1, int(self.genes['inteligencia'] * 64))
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(28,)),  # Tamaño del estado actualizado (número de campos en Estado)
            tf.keras.layers.Dense(inteligencia, activation='relu'),
            tf.keras.layers.Dense(inteligencia // 2, activation='relu'),
            tf.keras.layers.Dense(8, activation='linear')
        ])
        modelo.compile(optimizer='adam', loss='mse')
        return modelo

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
        
        return np.array(estado)

    def actualizar(self, mundo):
        self.edad += 1
        self.hambre += 1
        self.sed += 1
        
        # Reducir energía basada en el metabolismo
        self.energia = max(config.MIN_ENERGIA, self.energia - 0.5 * self.genes['metabolismo'])
        self.agregar_log("Actualizar", f"Energía: {self.energia}")

        # Actualizar epsilon para la exploración
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        # Verificar si la entidad muere por hambre o sed
        if self.hambre >= config.MAX_HAMBRE or self.sed >= config.MAX_SED:
            self.energia = config.MIN_ENERGIA
            self.agregar_log("Actualizar", f"Murió por hambre o sed")
        # Tomar una decisión y ejecutar una acción
        estado = self.obtener_estado(mundo)
        decision = self.tomar_decision(estado)
        self.agregar_log("Actualizar", f"Decision: {decision}")
        recompensa = self.ejecutar_accion(decision, mundo)
        self.agregar_log("Actualizar", f"Recompensa: {recompensa}")
        
        # Actualizar el buffer de experiencia
        nuevo_estado = self.obtener_estado(mundo)
        self.buffer_experiencia.append((estado, decision, recompensa, nuevo_estado))
        
        # Entrenar la red neuronal
        if len(self.buffer_experiencia) >= config.TAMANO_MINIBATCH:
            self.entrenar()
            self.agregar_log("Actualizar", f"Entrenamiento: {self.entrenar()}")
        # Actualizar historial de recompensas
        self.historial_recompensas.append(recompensa)
        if len(self.historial_recompensas) > 100:
            self.historial_recompensas.pop(0)

        # Calcular cambios en puntuación y energía
        self.cambio_puntuacion = self.puntuacion - (self.historial_puntuacion[-1] if self.historial_puntuacion else self.puntuacion)
        self.cambio_energia = self.energia - (self.historial_energia[-1] if self.historial_energia else self.energia)

        # Actualizar historiales
        self.historial_puntuacion.append(self.puntuacion)
        self.historial_energia.append(self.energia)

    def tomar_decision(self, estado):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(6)
        q_valores = self.red_neuronal.predict(np.array([estado]))[0]
        return np.argmax(q_valores)

    def entrenar(self):
        minibatch = random.sample(self.buffer_experiencia, config.TAMANO_MINIBATCH)
        estados = np.array([estado for estado, _, _, _ in minibatch])
        
        nuevos_estados = np.array([nuevo_estado for _, _, _, nuevo_estado in minibatch])
        
        q_valores = self.red_neuronal.predict(estados)
        q_valores_siguientes = self.red_neuronal.predict(nuevos_estados)
        
        x = []
        y = []
        for i, (estado, accion, recompensa, _) in enumerate(minibatch):
            if recompensa == -10:  # Si la entidad murió
                target = recompensa
            else:
                target = recompensa + config.FACTOR_DESCUENTO * np.amax(q_valores_siguientes[i])
            
            target_f = q_valores[i]
            target_f[accion] = target
            x.append(estado)
            y.append(target_f)
        
        historia = self.red_neuronal.fit(np.array(x), np.array(y), epochs=1, verbose=0)
        self.historial_perdidas.append(historia.history['loss'][0])
        if len(self.historial_perdidas) > 100:
            self.historial_perdidas.pop(0)
        
        return historia.history['loss'][0]

    def ejecutar_accion(self, decision, mundo):
        recompensa = 0
        consumo_energia = 1 * self.genes['metabolismo']
        
        if decision < 4:  # Movimiento
            if decision == 0:  # Mover arriba
                self.posicion_y = min(config.MAX_Y, self.posicion_y + 1 * self.genes['velocidad'])
                self.agregar_log("Actualizar", f"Mover arriba: {self.posicion_y}")
            elif decision == 1:  # Mover abajo
                self.posicion_y = max(config.MIN_Y, self.posicion_y - 1 * self.genes['velocidad'])
                self.agregar_log("Actualizar", f"Mover abajo: {self.posicion_y}")
            elif decision == 2:  # Mover izquierda
                self.posicion_x = max(config.MIN_X, self.posicion_x - 1 * self.genes['velocidad'])
                self.agregar_log("Actualizar", f"Mover izquierda: {self.posicion_x}")
            elif decision == 3:  # Mover derecha
                self.posicion_x = min(config.MAX_X, self.posicion_x + 1 * self.genes['velocidad'])
                self.agregar_log("Actualizar", f"Mover derecha: {self.posicion_x}")
            consumo_energia *= config.MULTIPLICADOR_CONSUMO_MOVIMIENTO
            self.agregar_log("Actualizar", f"Consumo de energía: {consumo_energia}")
        elif decision == 4:  # Comer
            comida_consumida = mundo.consumir_recurso(x=self.posicion_x, y=self.posicion_y, tipo_recurso='comida', nombre_entidad=self.nombre)
            self.energia = min(config.MAX_ENERGIA, self.energia + comida_consumida * config.ENERGIA_POR_COMIDA)
            self.hambre = max(config.MIN_HAMBRE, self.hambre - comida_consumida * config.REDUCCION_HAMBRE_POR_COMIDA)
            recompensa += comida_consumida * config.RECOMPENSA_COMIDA
            self.agregar_log("Actualizar", f"Comer: {comida_consumida}, Energía: {self.energia}, Hambre: {self.hambre}, Recompensa: {recompensa}") 
        elif decision == 5:  # Beber
            agua_consumida = mundo.consumir_recurso(x=self.posicion_x, y=self.posicion_y, tipo_recurso='agua', nombre_entidad=self.nombre)
            self.energia = min(config.MAX_ENERGIA, self.energia + agua_consumida * config.ENERGIA_POR_AGUA)
            self.sed = max(config.MIN_SED, self.sed - agua_consumida * config.REDUCCION_SED_POR_AGUA)
            recompensa += agua_consumida * config.RECOMPENSA_AGUA
            self.agregar_log("Actualizar", f"Beber: {agua_consumida}, Energía: {self.energia}, Sed: {self.sed}, Recompensa: {recompensa}")
        elif decision == 6:  # Atacar
            entidad_cercana = mundo.obtener_entidad_mas_cercana(self.posicion_x, self.posicion_y)
            if entidad_cercana:
                daño = self.genes['agresividad'] * config.ENERGIA_ATAQUE
                entidad_cercana.energia = max(config.MIN_ENERGIA, entidad_cercana.energia - daño)
                self.energia = min(config.MAX_ENERGIA, self.energia + daño / 2)
                recompensa += config.RECOMPENSA_ATAQUE
                consumo_energia *= config.MULTIPLICADOR_CONSUMO_ATAQUE
                self.agregar_log("Actualizar", f"Atacar: {daño}, Energía: {self.energia}, Recompensa: {recompensa}")
            else:
                recompensa = config.PENALIZACION_ACCION_FALLIDA
        elif decision == 7:  # Socializar
            entidad_cercana = mundo.obtener_entidad_mas_cercana(self.posicion_x, self.posicion_y)
            if entidad_cercana:
                beneficio = self.genes['sociabilidad'] * config.ENERGIA_SOCIALIZACION
                self.energia = min(config.MAX_ENERGIA, self.energia + beneficio)
                entidad_cercana.energia = min(config.MAX_ENERGIA, entidad_cercana.energia + beneficio)
                recompensa += config.RECOMPENSA_SOCIALIZACION
                consumo_energia *= config.MULTIPLICADOR_CONSUMO_SOCIALIZACION
                self.agregar_log("Actualizar", f"Socializar: {beneficio}, Energía: {self.energia}, Recompensa: {recompensa}")
            else:
                recompensa = config.PENALIZACION_ACCION_FALLIDA
        
        self.energia = max(config.MIN_ENERGIA, self.energia - consumo_energia)
        self.agregar_log("Actualizar", f"Energía: {self.energia}, Consumo de energía: {consumo_energia}")
        self.acciones_tomadas[decision] += 1
        self.agregar_log("Actualizar", f"Acciones tomadas: {self.acciones_tomadas}")
        
        if self.energia <= config.MIN_ENERGIA:
            recompensa = config.PENALIZACION_ENERGIA_BAJA
            self.agregar_log("Actualizar", f"Recompensa: {recompensa}, Energía: {self.energia}")
        elif self.energia > config.UMBRAL_ENERGIA_ALTA:
            recompensa += config.RECOMPENSA_ENERGIA_ALTA
            self.agregar_log("Actualizar", f"Recompensa: {recompensa}, Energía: {self.energia}")

        # Actualizar la puntuación de la entidad
        self.puntuacion += recompensa
        self.agregar_log("Actualizar", f"Puntuación: {self.puntuacion}")

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

    def reproducir(self, pareja):
        self.agregar_log("Reproducir", f"Reproducir: {self.nombre} y {pareja.nombre}")
        nuevos_genes = {}
        for gen in self.genes:
            if random.random() < 0.5:
                nuevos_genes[gen] = self.genes[gen]
            else:
                nuevos_genes[gen] = pareja.genes[gen]
            # Mutación
            if random.random() < config.PROBABILIDAD_MUTACION:
                nuevos_genes[gen] *= random.uniform(config.RANGO_MUTACION_MIN, config.RANGO_MUTACION_MAX)
        
        nueva_entidad = EntidadIA(f"Hijo de {self.nombre} y {pareja.nombre}", 
                                  (self.posicion_x + pareja.posicion_x) / 2,
                                  (self.posicion_y + pareja.posicion_y) / 2,
                                  config.ENERGIA_INICIAL, nuevos_genes)
        nueva_entidad.generacion = max(self.generacion, pareja.generacion) + 1
        
        # Transferencia de conocimiento
        try:
            pesos_padre = self.ajustar_pesos(self.red_neuronal.get_weights(), nueva_entidad.red_neuronal.get_weights())
            pesos_madre = self.ajustar_pesos(pareja.red_neuronal.get_weights(), nueva_entidad.red_neuronal.get_weights())
            
            nueva_entidad.red_neuronal.set_weights([
                (pesos_padre[i] + pesos_madre[i]) / 2
                for i in range(len(pesos_padre))
            ])
        except Exception as e:
            print(f"Error al transferir conocimiento: {e}")
            # En caso de error, mantenemos la red neuronal original de la nueva entidad
            pass

        self.agregar_log("Reproducir", f"Nueva entidad: {nueva_entidad.nombre}")
        return nueva_entidad

    def obtener_metricas_aprendizaje(self):
        return {
            "recompensa_promedio": np.mean(self.historial_recompensas) if self.historial_recompensas else 0,
            "perdida_promedio": np.mean(self.historial_perdidas) if self.historial_perdidas else 0,
            "epsilon": self.epsilon
        }

    def to_dict(self):
        metricas = self.obtener_metricas_aprendizaje()
        return {
            "nombre": self.nombre,
            "posicion_x": self.posicion_x,
            "posicion_y": self.posicion_y,
            "energia": self.energia,
            "puntuacion": self.puntuacion,
            "recompensa_promedio": metricas["recompensa_promedio"],
            "perdida_promedio": metricas["perdida_promedio"],
            "epsilon": metricas["epsilon"],
            "genes": self.genes,
            "generacion": self.generacion,
            "acciones_tomadas": self.acciones_tomadas,
            "edad": self.edad,
            "hambre": self.hambre,
            "sed": self.sed,
            "cambio_puntuacion": self.cambio_puntuacion,
            "cambio_energia": self.cambio_energia,
            "logs": [log.to_dict() for log in self.logs]
        }

    def esta_lista_para_reproducirse(self):
        return self.edad >= config.EDAD_REPRODUCCION and self.energia > config.ENERGIA_REPRODUCCION and self.puntuacion > config.PUNTUACION_REPRODUCCION