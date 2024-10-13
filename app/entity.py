import numpy as np
import tensorflow as tf
import logging
import random

logger = logging.getLogger(__name__)

class Entity:
    def __init__(self, config):
        self.config = config
        self.id = random.randint(1, 1000)
        self.position = np.random.rand(2) * self.config.WORLD_SIZE
        self.energy = self.config.INITIAL_ENERGY
        self.age = 0
        self.neural_network = self.create_neural_network()

    def create_neural_network(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.config.INPUT_SIZE,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(self.config.OUTPUT_SIZE, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='mse')
        return model

    def update(self, world):
        old_energy = self.energy
        old_position = self.position.copy()
        
        state = self.get_state(world)
        action = self.decide_action(state)
        self.perform_action(action, world)
        self.energy -= self.config.ENERGY_CONSUMPTION
        self.age += 1
        
        reward = self.calculate_reward(old_energy, old_position, world)
        self.train(state, action, reward, self.get_state(world))

    def get_state(self, world):
        nearby_entities = self.get_nearby_entities(world)
        state = np.concatenate([
            self.position,  # 2 valores
            [self.energy],  # 1 valor
            [self.age],  # 1 valor
            [len(nearby_entities)],  # 1 valor
            np.mean([e.position for e in nearby_entities], axis=0) if nearby_entities else [0, 0],  # 2 valores
            [np.mean([e.energy for e in nearby_entities]) if nearby_entities else 0],  # 1 valor
            [world.get_food_at(self.position)]  # 1 valor
        ])
        return state  # Total: 9 valores

    def decide_action(self, state):
        if np.random.rand() < self.config.EPSILON:
            return np.random.rand(self.config.OUTPUT_SIZE)
        else:
            return self.neural_network.predict(state.reshape(1, -1))[0]

    def perform_action(self, action, world):
        move_x, move_y, eat, reproduce, attack, defend = action
        
        # Movimiento
        self.position[0] = (self.position[0] + move_x * self.config.MAX_SPEED) % self.config.WORLD_SIZE
        self.position[1] = (self.position[1] + move_y * self.config.MAX_SPEED) % self.config.WORLD_SIZE
        
        # Comer
        if eat > 0.5:
            logger.info(f"Entity {self.id} is eating")
            food = world.get_food_at(self.position)
            self.energy += food
            world.consume_food_at(self.position)
        
        # Reproducirse
        if reproduce > 0.5 and self.energy > self.config.REPRODUCTION_ENERGY:
            logger.info(f"Entity {self.id} is reproducing")
            world.add_entity(self.reproduce())
        
        # Atacar y defender se implementarán en la lógica del mundo

    def get_nearby_entities(self, world):
        return [e for e in world.entities if np.linalg.norm(e.position - self.position) < self.config.VISION_RANGE and e != self]

    def reproduce(self):
        child = Entity(self.config)
        child.neural_network.set_weights([w + np.random.normal(0, 0.1, w.shape) for w in self.neural_network.get_weights()])
        child.energy = self.energy / 2
        self.energy /= 2
        return child

    def to_dict(self):
        return {
            "id": self.id,
            "position": self.position.tolist(),
            "energy": self.energy,
            "age": self.age
        }

    def calculate_reward(self, old_energy, old_position, world):
        reward = 0
        
        # Recompensa por obtener energía
        energy_diff = self.energy - old_energy
        reward += energy_diff * 0.1
        
        # Recompensa por moverse hacia la comida
        old_food = world.get_food_at(old_position)
        new_food = world.get_food_at(self.position)
        if new_food > old_food:
            reward += 0.5
        
        # Penalización por morir
        if self.energy <= 0:
            reward -= 10
        
        return reward

    def train(self, state, action, reward, next_state):
        target = reward + 0.99 * np.max(self.neural_network.predict(next_state.reshape(1, -1))[0])
        target_vec = self.neural_network.predict(state.reshape(1, -1))[0]
        target_vec[np.argmax(action)] = target
        self.neural_network.fit(state.reshape(1, -1), target_vec.reshape(1, -1), epochs=1, verbose=0)
