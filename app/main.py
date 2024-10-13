import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from app.world import World

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.railway.app", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

world = World()

@app.on_event("startup")
async def startup_event():
    await world.initialize()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    world.add_websocket_connection(websocket)
    try:
        while True:
            await websocket.receive_text()  # Mantener la conexión abierta
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        world.remove_websocket_connection(websocket)
        await websocket.close()

@app.get("/state")
async def get_state():
    return world.get_state()

@app.post("/start")
async def start_simulation():
    await world.start()
    return {"message": "Simulation started"}

@app.post("/stop")
async def stop_simulation():
    await world.stop()
    return {"message": "Simulation stopped"}

@app.post("/reset")
async def reset_simulation():
    await world.reset()
    return {"message": "Simulation reset"}