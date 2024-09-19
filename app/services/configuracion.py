class Configuracion:
    def __init__(self):
        # Configuración del mundo
        self.ANCHO_MUNDO = 100
        self.ALTO_MUNDO = 100
        self.TIEMPO_REPRODUCCION = 50
        self.NUMERO_INICIAL_ENTIDADES = 30
        self.NUMERO_INICIAL_RECURSOS = 150
        self.INTERVALO_ACTUALIZACION = 1  # en segundos

        # Límites del mundo
        self.MIN_X = -200
        self.MAX_X = 200
        self.MIN_Y = -200
        self.MAX_Y = 200

        # Configuración de entidades
        self.ENERGIA_INICIAL = 100
        self.EDAD_MAXIMA = 100
        self.ENERGIA_REPRODUCCION = 60
        self.PUNTUACION_REPRODUCCION = 100
        self.EDAD_REPRODUCCION = 30
        self.COSTO_ENERGIA_REPRODUCCION = 30
        self.DISTANCIA_MOVIMIENTO = 2
        self.TASA_APRENDIZAJE_BASE = 0.01
        self.FACTOR_DESCUENTO_BASE = 0.95
    
        # Límites de atributos de entidades
        self.MIN_ENERGIA = 0
        self.MAX_ENERGIA = 100
        self.MIN_HAMBRE = 0
        self.MAX_HAMBRE = 100
        self.MIN_SED = 0
        self.MAX_SED = 100
        self.MIN_EDAD = 0
        self.MAX_EDAD = 100
        self.MIN_GENERACION = 1
        self.MAX_GENERACION = 100
        self.MIN_CAMBIO_PUNTUACION = -100
        self.MAX_CAMBIO_PUNTUACION = 100
        self.MIN_CAMBIO_ENERGIA = -100
        self.MAX_CAMBIO_ENERGIA = 100

        # Configuración de genes
        self.MIN_VELOCIDAD = 0.5
        self.MAX_VELOCIDAD = 5.0
        self.MIN_VISION = 30
        self.MAX_VISION = 100
        self.MIN_METABOLISMO = 0.8
        self.MAX_METABOLISMO = 1.2
        self.MIN_AGRESIVIDAD = 0
        self.MAX_AGRESIVIDAD = 1
        self.MIN_SOCIABILIDAD = 0
        self.MAX_SOCIABILIDAD = 1
        self.MIN_INTELIGENCIA = 0.5
        self.MAX_INTELIGENCIA = 1.5
        self.MIN_RESISTENCIA = 0.5
        self.MAX_RESISTENCIA = 1.5
        self.MIN_ADAPTABILIDAD = 0.5
        self.MAX_ADAPTABILIDAD = 1.5

        # Configuración adicional de entidades
        self.DISTANCIA_INTERACCION = 2
        self.ENERGIA_ATAQUE = 10
        self.ENERGIA_SOCIALIZACION = 1
        self.UMBRAL_ENERGIA_ALTA = 80
        self.RECOMPENSA_ENERGIA_ALTA = 1
        self.PENALIZACION_ENERGIA_BAJA = -10
        self.RECOMPENSA_COMIDA = 12
        self.RECOMPENSA_AGUA = 9
        self.RECOMPENSA_ATAQUE = 3
        self.RECOMPENSA_SOCIALIZACION = 3
        self.PENALIZACION_ACCION_FALLIDA = -5
        self.ENERGIA_POR_COMIDA = 40
        self.ENERGIA_POR_AGUA = 20
        self.REDUCCION_HAMBRE_POR_COMIDA = 50
        self.REDUCCION_SED_POR_AGUA = 50
        self.MULTIPLICADOR_CONSUMO_MOVIMIENTO = 1
        self.MULTIPLICADOR_CONSUMO_ATAQUE = 1
        self.MULTIPLICADOR_CONSUMO_SOCIALIZACION = 1
        self.PROBABILIDAD_MUTACION = 0.1
        self.RANGO_MUTACION_MIN = 0.9
        self.RANGO_MUTACION_MAX = 1.1
        self.RECOMPENSA_REPRODUCCION = 10

        # Configuración de recursos
        self.MIN_RECURSOS = 0
        self.MAX_RECURSOS = 100
        self.DISTANCIA_VISION_RECURSOS = 100
        self.INTERVALO_REGENERACION_RECURSOS = 10  # Cada cuántas actualizaciones se regeneran los recursos

        # Configuración de clima
        self.MIN_TEMPERATURA = -30
        self.MAX_TEMPERATURA = 50
        self.MIN_TIEMPO_DEL_DIA = 0
        self.MAX_TIEMPO_DEL_DIA = 1
        self.MIN_PELIGRO = 0
        self.MAX_PELIGRO = 1

        # Configuración de mundo
        self.CICLO_DIA = 24

        # Configuración de IA
        self.TAMANO_BUFFER_EXPERIENCIA = 2000
        self.TAMANO_MINIBATCH = 32
        self.EPSILON_INICIAL = 0.5
        self.EPSILON_MINIMO = 0.01
        self.EPSILON_DECAY = 0.995
        self.FACTOR_DESCUENTO = 0.95

        # Logs
        self.MAX_LOGS_POR_ENTIDAD = 100

config = Configuracion()