from fastapi import FastAPI
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