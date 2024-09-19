from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.database import engine, Base
from app.services.mundo import Mundo
import asyncio
import json
import logging
from fastapi.middleware.cors import CORSMiddleware

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

Base.metadata.create_all(bind=engine)

mundo = Mundo()

@app.on_event("startup")
async def startup_event():
    mundo.iniciar()
    print("El mundo ha sido iniciado y poblado con entidades.")

@app.get("/")
async def root():
    return {"mensaje": "Servidor de mundo persistente en funcionamiento"}

@app.get("/estado")
async def obtener_estado():
    return mundo.obtener_estado()

@app.get("/visualizacion", response_class=HTMLResponse)
async def visualizacion():
    with open("static/index.html", "r") as file:
        return file.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        estado = mundo.obtener_estado()
        await websocket.send_text(json.dumps(estado))
        await asyncio.sleep(1)  # Enviar actualizaciones cada segundo

@app.post("/entidad")
async def crear_entidad(nombre: str, x: int, y: int):
    try:
        nueva_entidad = mundo.crear_entidad(nombre, x, y)
        return {"mensaje": f"Entidad {nombre} creada en ({x}, {y})"}
    except Exception as e:
        logger.error(f"Error al crear entidad: {str(e)}")
        return {"error": str(e)}

@app.post("/pausar")
async def pausar_simulacion():
    mundo.pausar()
    return {"mensaje": "Simulación pausada"}

@app.post("/reanudar")
async def reanudar_simulacion():
    mundo.reanudar()
    return {"mensaje": "Simulación reanudada"}

@app.post("/reiniciar")
async def reiniciar_simulacion():
    mundo.reiniciar()
    return {"mensaje": "Simulación reiniciada"}