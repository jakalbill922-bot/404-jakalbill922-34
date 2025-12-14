from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
import base64

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import pyspz

from config import settings
from logger_config import logger
from schemas import GenerateRequest, GenerateResponse
from modules import GenerationPipeline


pipeline = GenerationPipeline(settings)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await pipeline.startup()
    try:
        yield
    finally:
        await pipeline.shutdown()


app = FastAPI(
    title=settings.api_title,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ready"}

@app.post("/generate_from_base64", response_model=GenerateResponse)
async def generate_from_base64(request: GenerateRequest) -> GenerateResponse:
    try:
        result = await pipeline.generate_gs(request)

        compressed_ply_bytes = None

        if result.ply_file_base64 and settings.compression:
            compressed_ply_bytes = pyspz.compress(result.ply_file_base64, workers=1)
            logger.info(f"Compressed PLY size: {len(compressed_ply_bytes)} bytes")

        result.ply_file_base64 = base64.b64encode(result.ply_file_base64 if not compressed_ply_bytes else compressed_ply_bytes).decode("utf-8")

        return result

    except Exception as exc:
        logger.exception(f"Error generating task: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/generate")
async def generate(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> StreamingResponse:
    try:
        logger.info(f"Task received. Uploading image: {prompt_image_file.filename}")

        ply_bytes = await pipeline.generate_from_upload(await prompt_image_file.read(), seed)

        logger.info(f"Task completed. PLY size: {len(ply_bytes)} bytes")

        return StreamingResponse(
            BytesIO(ply_bytes),
            media_type="application/octet-stream",
        )

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

@app.post("/generate-spz")
async def generate_spz(prompt_image_file: UploadFile = File(...), seed: int = Form(-1)) -> StreamingResponse:
    try:
        logger.info(f"Task received (SPZ). Uploading image: {prompt_image_file.filename}")

        ply_bytes = await pipeline.generate_from_upload(await prompt_image_file.read(), seed)

        if ply_bytes:
            ply_compressed_bytes = pyspz.compress(ply_bytes, workers=1)
            logger.info(f"Task completed. SPZ size: {len(ply_compressed_bytes)} bytes")

        return StreamingResponse(
            BytesIO(ply_compressed_bytes),
            media_type="application/octet-stream",
        )

    except Exception as exc:
        logger.exception(f"Error generating from upload: {exc}")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/setup/info")
async def get_setup_info() -> dict:
    try:
        return settings.dict()
    except Exception as e:
        logger.error(f"Failed to get setup info: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve configuration")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "serve:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
