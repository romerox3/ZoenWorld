class Config:
    def __init__(self):
        self.WORLD_SIZE = 100
        self.INITIAL_ENTITIES = 10
        self.INITIAL_ENERGY = 100
        self.ENERGY_CONSUMPTION = 0.1
        self.UPDATE_INTERVAL = 0.1
        self.INPUT_SIZE = 9  # Aumentamos el tamaño de entrada
        self.OUTPUT_SIZE = 6  # Aumentamos el tamaño de salida
        self.VISION_RANGE = 10
        self.MAX_SPEED = 2
        self.REPRODUCTION_ENERGY = 80
        self.REPRODUCTION_THRESHOLD = 0.7
        self.EPSILON = 0.1
