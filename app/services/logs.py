from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from app.models import Log, Entidad
from app.schemas import LogCreate
from datetime import datetime, timezone
from app.database import AsyncSessionLocal

class LogService:
    async def crear_log(self, log: LogCreate):
        async with AsyncSessionLocal() as db:
            async with db.begin():
                try:
                    if log.entidad_id:
                        # Verificar si la entidad existe
                        entidad = await db.execute(select(Entidad).where(Entidad.id == log.entidad_id))
                        if not entidad.scalar_one_or_none():
                            print(f"Advertencia: Entidad con id {log.entidad_id} no encontrada. No se creará el log.")
                            return None

                    log.tiempo = datetime.now(timezone.utc)
                    nuevo_log = Log(**log.dict())
                    db.add(nuevo_log)
                    await db.flush()
                    return nuevo_log
                except Exception as e:
                    print(f"Error al crear log: {str(e)}")
                    raise

    async def obtener_logs(self, entidad_id: int = None):
        async with AsyncSessionLocal() as db:
            query = select(Log)
            if entidad_id:
                query = query.filter(Log.entidad_id == entidad_id)
            result = await db.execute(query)
            logs = result.scalars().all()
            if not logs:
                return []
            return [{'id': log.id, 'tiempo': log.tiempo, 'accion': log.accion, 'detalles': log.detalles, 'entidad_nombre': log.entidad.nombre} for log in logs]