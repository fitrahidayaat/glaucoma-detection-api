from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import random
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Next.js app's URL for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        # Simulate processing the uploaded image
        image_content = await image.read()

        # Simulated prediction logic
        risk = random.random()
        prediction = "High risk of glaucoma" if risk > 0.5 else "Low risk of glaucoma"
        confidence = round(risk * 100, 2)

        return JSONResponse(
            content={
                "prediction": prediction,
                "confidence": f"{confidence}"
            }
        )
    except Exception as e:
        return JSONResponse(
            content={"error": str(e)},
            status_code=500
        )
