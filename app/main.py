from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from app.database import engine, Base
from app.services.mundo import Mundo
import asyncio
import json

app = FastAPI()

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