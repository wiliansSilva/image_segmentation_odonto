from fastapi import FastAPI, File, UploadFile

from models.unet import load_unet_model

app = FastAPI()

@app.get("/")
async def home():
    return {
        "Message": "Hello from api"
    }

@app.get("/unet/predict-single")
def unet_predict(file: UploadFile):
    model = load_unet_model()
    preds = 