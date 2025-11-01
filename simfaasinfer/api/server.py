# simfaasinfer/api/server.py
"""
REST API server for submitting profiling and search jobs. FastAPI skeleton.
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import Any, Dict, Optional
import uuid
import time

app = FastAPI(title="SimFaaSInfer API", version="0.1.0")

# In-memory job storage (use Redis/DB in production)
jobs = {}


class ProfilingRequest(BaseModel):
    model_spec: Dict[str, Any]
    parallelism: Dict[str, Any]
    target_device: str
    output_dir: Optional[str] = "/tmp/profiles"


class SearchRequest(BaseModel):
    workload: Dict[str, Any]
    config_space: Dict[str, Any]
    constraints: Dict[str, Any]


class CalibrationRequest(BaseModel):
    estimator_path: str
    telemetry_samples: list


@app.get("/")
async def root():
    return {
        "service": "SimFaaSInfer API",
        "version": "0.1.0",
        "endpoints": ["/profiling/run", "/search", "/calibrate"]
    }


@app.post("/profiling/run")
async def run_profiling(req: ProfilingRequest, background_tasks: BackgroundTasks):
    """Start a profiling job."""
    job_id = f"profiling-{uuid.uuid4().hex[:8]}"
    
    # Create job
    jobs[job_id] = {
        "id": job_id,
        "type": "profiling",
        "status": "scheduled",
        "request": req.dict(),
        "created_at": time.time(),
        "result": None
    }
    
    # Schedule background task
    background_tasks.add_task(_run_profiling_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "scheduled",
        "message": "Profiling job scheduled"
    }


@app.post("/search")
async def run_search(req: SearchRequest, background_tasks: BackgroundTasks):
    """Start a capacity search job."""
    job_id = f"search-{uuid.uuid4().hex[:8]}"
    
    jobs[job_id] = {
        "id": job_id,
        "type": "search",
        "status": "scheduled",
        "request": req.dict(),
        "created_at": time.time(),
        "result": None
    }
    
    background_tasks.add_task(_run_search_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "scheduled",
        "message": "Search job scheduled"
    }


@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job status and results."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]


@app.get("/jobs")
async def list_jobs(job_type: Optional[str] = None, status: Optional[str] = None):
    """List all jobs with optional filtering."""
    filtered_jobs = []
    
    for job in jobs.values():
        if job_type and job['type'] != job_type:
            continue
        if status and job['status'] != status:
            continue
        filtered_jobs.append(job)
    
    return {"jobs": filtered_jobs, "count": len(filtered_jobs)}


@app.post("/calibrate")
async def run_calibration(req: CalibrationRequest, background_tasks: BackgroundTasks):
    """Start a calibration job."""
    job_id = f"calibrate-{uuid.uuid4().hex[:8]}"
    
    jobs[job_id] = {
        "id": job_id,
        "type": "calibration",
        "status": "scheduled",
        "request": req.dict(),
        "created_at": time.time(),
        "result": None
    }
    
    background_tasks.add_task(_run_calibration_job, job_id, req)
    
    return {
        "job_id": job_id,
        "status": "scheduled",
        "message": "Calibration job scheduled"
    }


# Background task functions
def _run_profiling_job(job_id: str, request: ProfilingRequest):
    """Execute profiling job."""
    try:
        jobs[job_id]["status"] = "running"
        
        from ..profiler.operator_triage import generate_profiling_plan
        from ..profiler.profiler_runner import run_profiles
        
        # Generate plan
        plan = generate_profiling_plan(
            request.model_spec,
            request.parallelism
        )
        
        # Run profiles
        artifacts = run_profiles(
            plan,
            request.target_device,
            request.output_dir or "/tmp/profiles"
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "num_artifacts": len(artifacts),
            "output_dir": request.output_dir
        }
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


def _run_search_job(job_id: str, request: SearchRequest):
    """Execute search job."""
    try:
        jobs[job_id]["status"] = "running"
        
        from ..optimizer.vidur_search import search_workload
        
        result = search_workload(
            request.workload,
            request.config_space,
            request.constraints
        )
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = result
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


def _run_calibration_job(job_id: str, request: CalibrationRequest):
    """Execute calibration job."""
    try:
        jobs[job_id]["status"] = "running"
        
        from ..runtime_estimator.rf_estimator import RFEstimator
        from ..calibration.calibrator import calibrate
        
        # Load estimator
        estimator = RFEstimator()
        estimator.load(request.estimator_path)
        
        # Calibrate
        calibrated = calibrate(estimator, request.telemetry_samples)
        
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["result"] = {
            "correction_models": calibrated.correction_models,
            "confidence_scores": calibrated.confidence_scores
        }
        
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)