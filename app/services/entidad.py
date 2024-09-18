import numpy as np
import tensorflow as tf
from collections import deque
import random
from app.models import Entidad as EntidadModel
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow logging
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
        self.buffer_experiencia = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.historial_recompensas = []
        self.historial_perdidas = []
        self.generacion = 1

    def generar_genes(self):
        return {
            'velocidad': random.uniform(0.5, 2.0),
            'vision': random.uniform(30, 100),
            'metabolismo': random.uniform(0.8, 1.2),
            'agresividad': random.uniform(0, 1),
            'sociabilidad': random.uniform(0, 1),
            'inteligencia': random.uniform(0.5, 1.5),
            'resistencia': random.uniform(0.5, 1.5),
            'adaptabilidad': random.uniform(0.5, 1.5)
        }

    def crear_red_neuronal(self):
        inteligencia = max(1, int(self.genes['inteligencia'] * 64))  # Aseguramos un mínimo de 1 neurona
        modelo = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(11,)),
            tf.keras.layers.Dense(inteligencia, activation='relu'),
            tf.keras.layers.Dense(inteligencia // 2, activation='relu'),
            tf.keras.layers.Dense(6, activation='linear')
        ])
        modelo.compile(optimizer='adam', loss='mse')
        return modelo

    def obtener_estado(self, mundo):
        recursos_cercanos = mundo.obtener_recursos_cercanos(self.posicion_x, self.posicion_y)
        entidades_cercanas = mundo.obtener_entidades_cercanas(self.posicion_x, self.posicion_y)
        return np.array([
            self.posicion_x, self.posicion_y, self.energia,
            recursos_cercanos['comida'] * self.genes['vision'],
            recursos_cercanos['agua'] * self.genes['vision'],
            recursos_cercanos['arboles'] * self.genes['vision'],
            entidades_cercanas['aliados'] * self.genes['sociabilidad'],
            entidades_cercanas['enemigos'] * self.genes['agresividad'],
            mundo.obtener_tiempo_del_dia(),
            mundo.obtener_temperatura(),
            mundo.obtener_peligro(self.posicion_x, self.posicion_y) * (2 - self.genes['resistencia'])
        ])

    def tomar_decision(self, mundo):
        estado = self.obtener_estado(mundo)
        self.memoria.append(estado)
        
        if np.random.rand() <= self.epsilon:
            return random.randrange(6)
        
        q_values = self.red_neuronal.predict(np.array([estado]))
        return np.argmax(q_values[0])

    def actualizar(self, mundo):
        estado_anterior = self.obtener_estado(mundo)
        decision = self.tomar_decision(mundo)
        recompensa = self.ejecutar_accion(decision, mundo)
        nuevo_estado = self.obtener_estado(mundo)
        
        self.buffer_experiencia.append((estado_anterior, decision, recompensa, nuevo_estado))
        
        self.puntuacion += recompensa
        
        if len(self.buffer_experiencia) >= 32:
            self.entrenar()
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.historial_recompensas.append(recompensa)
        if len(self.historial_recompensas) > 100:
            self.historial_recompensas.pop(0)

    def entrenar(self):
        minibatch = random.sample(self.buffer_experiencia, 32)
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
                target = recompensa + 0.95 * np.amax(q_valores_siguientes[i])
            
            target_f = q_valores[i]
            target_f[accion] = target
            x.append(estado)
            y.append(target_f)
        
        historia = self.red_neuronal.fit(np.array(x), np.array(y), epochs=1, verbose=0)
        self.historial_perdidas.append(historia.history['loss'][0])
        if len(self.historial_perdidas) > 100:
            self.historial_perdidas.pop(0)

    def ejecutar_accion(self, decision, mundo):
        recompensa = 0
        if decision == 0:  # Mover arriba
            self.posicion_y = min(250, self.posicion_y + 10 * self.genes['velocidad'])
            recompensa = -1 * self.genes['metabolismo']
        elif decision == 1:  # Mover abajo
            self.posicion_y = max(-250, self.posicion_y - 10 * self.genes['velocidad'])
            recompensa = -1 * self.genes['metabolismo']
        elif decision == 2:  # Mover izquierda
            self.posicion_x = max(-250, self.posicion_x - 10 * self.genes['velocidad'])
            recompensa = -1 * self.genes['metabolismo']
        elif decision == 3:  # Mover derecha
            self.posicion_x = min(250, self.posicion_x + 10 * self.genes['velocidad'])
            recompensa = -1 * self.genes['metabolismo']
        elif decision == 4:  # Comer
            recompensa = mundo.comer(self.posicion_x, self.posicion_y) * self.genes['metabolismo']
        elif decision == 5:  # Descansar
            self.energia = min(100, self.energia + 5 * self.genes['resistencia'])
            recompensa = 1 * self.genes['metabolismo']

        self.energia = max(0, self.energia - 1 * self.genes['metabolismo'])
        if self.energia == 0:
            recompensa = -10  # Penalización por quedarse sin energía

        # Ajustar recompensa basada en la temperatura del mundo
        temperatura = mundo.obtener_temperatura()
        recompensa *= self.genes['adaptabilidad'] * (1 - abs(temperatura - 20) / 20)

        return recompensa

    def reproducir(self, pareja):
        nuevos_genes = {}
        for gen in self.genes:
            if random.random() < 0.5:
                nuevos_genes[gen] = self.genes[gen]
            else:
                nuevos_genes[gen] = pareja.genes[gen]
            # Mutación
            if random.random() < 0.1:
                nuevos_genes[gen] *= random.uniform(0.9, 1.1)
        
        nueva_entidad = EntidadIA(f"Hijo de {self.nombre} y {pareja.nombre}", 
                                  (self.posicion_x + pareja.posicion_x) / 2, 
                                  (self.posicion_y + pareja.posicion_y) / 2, 
                                  100, nuevos_genes)
        nueva_entidad.generacion = max(self.generacion, pareja.generacion) + 1
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
            "acciones_tomadas": self.acciones_tomadas if hasattr(self, 'acciones_tomadas') else {},
            "edad": self.edad if hasattr(self, 'edad') else 0,
        }