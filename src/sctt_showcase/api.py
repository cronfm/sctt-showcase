"""Small, local-first REST API for the synthetic demonstration."""

from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from plotly.offline import get_plotlyjs
import torch

from . import __version__
from .data import list_samples, load_scene
from .model import DemoEngine
from .schemas import InferRequest
from .visualization import build_figure

app = FastAPI(
    title="SCTT Showcase API",
    version=__version__,
    description="Synthetic local-coordinate localization. Toy-trained model; no real-world accuracy claims.",
)
_inference_lock = Lock()
MAX_BODY_BYTES = 262_144


class LimitedBody:
    """Bound decoded request bytes, including requests without Content-Length."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            await self.app(scope, receive, send)
            return
        chunks = []
        size = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            size += len(message.get("body", b""))
            if size > MAX_BODY_BYTES:
                await JSONResponse({"detail": "Request body exceeds 256 KiB."}, status_code=413)(
                    scope, receive, send
                )
                return
            chunks.append(message.get("body", b""))
            if not message.get("more_body", False):
                break
        body = b"".join(chunks)
        sent = False

        async def replay():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay, send)


app.add_middleware(LimitedBody)


@app.exception_handler(RequestValidationError)
async def invalid_input(_request, exc):
    # Non-finite JSON numbers can appear in Pydantic's raw input/context.
    # Return useful field locations without echoing unencodable input values.
    errors = [{key: error[key] for key in ("type", "loc", "msg")} for error in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=422)


class FigureRequest(InferRequest):
    color_mode: Literal["rgb", "feature"] = "rgb"
    backend: Literal["plotly", "pytorch3d"] = "plotly"


@lru_cache(maxsize=1)
def get_engine() -> DemoEngine:
    torch.set_num_threads(2)
    return DemoEngine(device="cpu")


def run_inference(request: InferRequest) -> dict:
    try:
        with _inference_lock:
            return get_engine().infer(request)
    except KeyError:
        raise HTTPException(404, "Unknown sample ID. See /api/samples.") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None


@app.get("/api/health", tags=["Discovery"])
def health():
    return {"status": "ok", "version": __version__, "synthetic": True, "device": "cpu"}


@app.get("/api/capabilities", tags=["Discovery"])
def capabilities():
    try:
        from pytorch3d.structures import Pointclouds  # noqa: F401
        from pytorch3d.transforms import quaternion_to_matrix  # noqa: F401

        available = True
    except (ImportError, OSError):
        available = False
    return {"plotly": True, "pytorch3d": available, "synthetic": True}


@app.get("/api/scene", tags=["Sample data"])
def scene():
    return load_scene()


@app.get("/api/samples", tags=["Sample data"])
def samples():
    return list_samples()


@app.post("/api/infer", tags=["Inference"])
def infer(request: InferRequest):
    return run_inference(request)


@app.post("/api/figure", tags=["Visualization"])
def figure(request: FigureRequest):
    result = run_inference(request)
    try:
        chart = build_figure(load_scene(), result, color_mode=request.color_mode, backend=request.backend)
    except (ImportError, OSError):
        raise HTTPException(
            503, "PyTorch3D is unavailable. Follow docs/pytorch3d.md or select Plotly."
        ) from None
    return Response(chart.to_json(), media_type="application/json")


@app.get("/api/example", tags=["Sample data"])
def example():
    """Return a minimal request that can be posted directly to /api/infer."""
    return {"sample_id": "loop-01", "top_k": 4, "candidate_offset_m": [0.0, 0.0, 0.0]}


@lru_cache(maxsize=1)
def plotly_js() -> str:
    return get_plotlyjs()


@app.get("/vendor/plotly.min.js", include_in_schema=False)
def plotly_bundle():
    return Response(
        plotly_js(), media_type="application/javascript", headers={"Cache-Control": "public, max-age=3600"}
    )


_static = Path(__file__).parent / "static"
if not _static.is_dir():
    _static = Path(__file__).parents[2] / "static"
app.mount("/static", StaticFiles(directory=_static), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(_static / "index.html")
