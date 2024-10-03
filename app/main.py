from fastapi import FastAPI, WebSocket, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.database import engine, Base, get_db
from app.services.mundo import Mundo
from app.services.logs import LogService
import asyncio
import json
import logging
from fastapi.middleware.cors import CORSMiddleware
from app.schemas import LogCreate
from sqlalchemy.orm import Session
from app.database import AsyncSessionLocal

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

@app.on_event("startup")
async def startup_event():
    global mundo
    await create_tables()
    async with AsyncSessionLocal() as db:
        mundo = Mundo(db)
        await mundo.iniciar()
    print("El mundo ha sido iniciado y poblado con entidades.")

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

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
    await mundo.pausar()
    return {"mensaje": "Simulación pausada"}

@app.post("/reanudar")
async def reanudar_simulacion():
    await mundo.reanudar()
    return {"mensaje": "Simulación reanudada"}

@app.post("/reiniciar")
async def reiniciar_simulacion():
    await mundo.reiniciar()
    return {"mensaje": "Simulación reiniciada"}

@app.post("/logs")
async def crear_log(log: LogCreate):
    return await mundo.log_service.crear_log(log)

@app.get("/logs")
async def obtener_logs(entidad_id: int = None):
    return await mundo.log_service.obtener_logs(entidad_id)