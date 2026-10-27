import datetime
import threading
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field
from ai_dj.errors import DJError, ResourceNotFoundError


class JobLogEntry(BaseModel):
    timestamp: str
    message: str
    level: str = "INFO"


class JobStatus(BaseModel):
    job_id: str
    name: str
    status: str  # "pending", "running", "completed", "failed", "cancelled"
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    created_at: str
    finished_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Dict[str, Any]] = None
    logs: List[JobLogEntry] = Field(default_factory=list)


class JobManager:
    def __init__(self, max_workers: int = 4):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.jobs: Dict[str, JobStatus] = {}
        self._futures: Dict[str, Future] = {}
        self._lock = threading.Lock()

    def submit_job(self, name: str, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> str:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.datetime.now(datetime.timezone.utc).isoformat()
        job = JobStatus(
            job_id=job_id,
            name=name,
            status="pending",
            progress=0.0,
            created_at=now,
        )
        with self._lock:
            self.jobs[job_id] = job

        def runner():
            job.status = "running"
            self.log(job_id, f"Job '{name}' started.")
            try:
                res = fn(*args, **kwargs)
                job.status = "completed"
                job.progress = 100.0
                job.result = res if isinstance(res, dict) else {"result": res}
                job.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.log(job_id, f"Job '{name}' completed successfully.")
            except Exception as e:
                job.status = "failed"
                job.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                if isinstance(e, DJError):
                    job.error = e.to_dict()
                else:
                    job.error = {"code": "EXECUTION_ERROR", "message": str(e), "recoverable": False}
                self.log(job_id, f"Job failed: {str(e)}", level="ERROR")

        fut = self.executor.submit(runner)
        self._futures[job_id] = fut
        return job_id

    def log(self, job_id: str, message: str, level: str = "INFO") -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                now = datetime.datetime.now(datetime.timezone.utc).isoformat()
                job.logs.append(JobLogEntry(timestamp=now, message=message, level=level))

    def update_progress(self, job_id: str, progress: float) -> None:
        with self._lock:
            job = self.jobs.get(job_id)
            if job:
                job.progress = max(0.0, min(100.0, float(progress)))

    def get_job(self, job_id: str) -> JobStatus:
        with self._lock:
            job = self.jobs.get(job_id)
            if not job:
                raise ResourceNotFoundError("job", job_id, suggested_tool="job_status")
            return job

    def cancel_job(self, job_id: str) -> bool:
        with self._lock:
            fut = self._futures.get(job_id)
            job = self.jobs.get(job_id)
            if not job:
                raise ResourceNotFoundError("job", job_id)
            if fut and fut.cancel():
                job.status = "cancelled"
                job.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                self.log(job_id, "Job cancelled by user request.")
                return True
            if job.status == "running":
                # Marked as cancelled
                job.status = "cancelled"
                job.finished_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
                return True
            return False
