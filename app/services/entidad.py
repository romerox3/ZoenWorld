import numpy as np
import tensorflow as tf
from collections import deque
import random
from app.models import Entidad as EntidadModel

class EntidadIA:
    def __init__(self, nombre, posicion_x, posicion_y, energia):
        self.nombre = nombre
        self.posicion_x = posicion_x
        self.posicion_y = posicion_y
        self.energia = energia
        self.red_neuronal = self.crear_red_neuronal()
        self.memoria = deque(maxlen=5)
        self.puntuacion = 0
        self.buffer_experiencia = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.historial_recompensas = []
        self.historial_perdidas = []

    def crear_red_neuronal(self):
        modelo = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(11,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(6, activation='linear')
        ])
        modelo.compile(optimizer='adam', loss='mse')
        return modelo

    def obtener_estado(self, mundo):
        recursos_cercanos = mundo.obtener_recursos_cercanos(self.posicion_x, self.posicion_y)
        entidades_cercanas = mundo.obtener_entidades_cercanas(self.posicion_x, self.posicion_y)
        return np.array([
            self.posicion_x, self.posicion_y, self.energia,
            recursos_cercanos['comida'], recursos_cercanos['agua'], recursos_cercanos['arboles'],
            entidades_cercanas['aliados'], entidades_cercanas['enemigos'],
            mundo.obtener_tiempo_del_dia(),
            mundo.obtener_temperatura(),
            mundo.obtener_peligro(self.posicion_x, self.posicion_y)
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
            self.posicion_y = min(250, self.posicion_y + 10)
            recompensa = -1
        elif decision == 1:  # Mover abajo
            self.posicion_y = max(-250, self.posicion_y - 10)
            recompensa = -1
        elif decision == 2:  # Mover izquierda
            self.posicion_x = max(-250, self.posicion_x - 10)
            recompensa = -1
        elif decision == 3:  # Mover derecha
            self.posicion_x = min(250, self.posicion_x + 10)
            recompensa = -1
        elif decision == 4:  # Comer
            recompensa = mundo.comer(self.posicion_x, self.posicion_y)
        elif decision == 5:  # Descansar
            self.energia = min(100, self.energia + 5)
            recompensa = 1

        self.energia = max(0, self.energia - 1)
        if self.energia == 0:
            recompensa = -10  # Penalización por quedarse sin energía

        return recompensa

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
            "epsilon": metricas["epsilon"]
        }