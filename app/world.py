import asyncio
from app.entity import Entity
from app.config import Config

class World:
    def __init__(self):
        self.entities = []
        self.running = False
        self.config = Config()

    async def initialize(self):
        for _ in range(self.config.INITIAL_ENTITIES):
            self.entities.append(Entity(self.config))

    async def start(self):
        self.running = True
        asyncio.create_task(self.main_loop())

    async def stop(self):
        self.running = False

    async def reset(self):
        self.entities = []
        await self.initialize()

    async def main_loop(self):
        while self.running:
            for entity in self.entities:
                entity.update(self)
            await asyncio.sleep(self.config.UPDATE_INTERVAL)

    def get_state(self):
        return {
            "entities": [entity.to_dict() for entity in self.entities],
            "running": self.running
        }