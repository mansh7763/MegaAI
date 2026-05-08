from celery import Celery
from api.config import get_settings

settings = get_settings()

celery_app = Celery(
    "mega_ai",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["api.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_soft_time_limit=300,
    task_time_limit=600,
)
