from fastapi import FastAPI, File
from uvicorn import run as app_run
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
from src.pipeline.train_pipeline import TrainPipeline
from src.pipeline.prediction_pipeline import ModelPredictor

from src.constant import *


app = FastAPI()

origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/train")
async def training():
    try:
        train_pipeline = TrainPipeline()

        train_pipeline.run_pipeline()

        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")


@app.post("/predict")
async def prediction(image_file: bytes = File(description="A file read as bytes")):
    try:
        prediction_pipeline = ModelPredictor()

        caption = prediction_pipeline.run_pipeline(image_file)

        result = {
            "caption" : caption
        }

        return JSONResponse(content=result, status_code=200)

    except Exception as e:
        JSONResponse(content = f"Error Occurred! {e}", status_code=500)

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)