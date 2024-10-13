import asyncio
import numpy as np
from app.entity import Entity
from app.config import Config
from fastapi import WebSocket
import logging

logger = logging.getLogger(__name__)

class World:
    def __init__(self):
        self.entities = []
        self.running = False
        self.config = Config()
        self.food_map = np.zeros((self.config.WORLD_SIZE, self.config.WORLD_SIZE))
        self.websocket_connections = []

    async def initialize(self):
        for _ in range(self.config.INITIAL_ENTITIES):
            self.entities.append(Entity(self.config))
        self.generate_food()

    async def emit_update(self):
        state = self.get_state()
        for connection in self.websocket_connections:
            await connection.send_json(state)

    def add_websocket_connection(self, connection):
        self.websocket_connections.append(connection)

    def remove_websocket_connection(self, connection):
        self.websocket_connections.remove(connection)

    async def start(self):
        self.running = True
        asyncio.create_task(self.main_loop())

    async def stop(self):
        self.running = False

    async def reset(self):
        self.entities = []
        self.food_map = np.zeros((self.config.WORLD_SIZE, self.config.WORLD_SIZE))
        await self.initialize()

    async def main_loop(self):
        while self.running:
            for entity in self.entities:
                entity.update(self)
            self.handle_interactions()
            self.remove_dead_entities()
            self.generate_food()
            await self.emit_update()
            await asyncio.sleep(self.config.UPDATE_INTERVAL)

    def handle_interactions(self):
        for i, entity in enumerate(self.entities):
            nearby_entities = entity.get_nearby_entities(self)
            for other in nearby_entities:
                if entity.energy > other.energy:
                    energy_diff = min(other.energy, entity.energy * 0.1)
                    entity.energy += energy_diff
                    other.energy -= energy_diff

    def remove_dead_entities(self):
        self.entities = [e for e in self.entities if e.energy > 0]

    def generate_food(self):
        new_food = np.random.rand(self.config.WORLD_SIZE, self.config.WORLD_SIZE) < 0.01
        self.food_map = np.minimum(self.food_map + new_food, 1)

    def get_food_at(self, position):
        x, y = int(position[0]), int(position[1])
        return self.food_map[x, y]

    def consume_food_at(self, position):
        x, y = int(position[0]), int(position[1])
        self.food_map[x, y] = 0

    def add_entity(self, entity):
        self.entities.append(entity)

    def get_state(self):
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "food_map": self.food_map.tolist(),
            "running": self.running
        }
