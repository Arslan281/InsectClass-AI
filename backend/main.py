
from io import BytesIO

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.predictor import analyze_image, class_names

# -----------------------------------
# 1. CREATE FASTAPI APPLICATION
# -----------------------------------

app = FastAPI(
    title="InsectClass AI API",
    description=(
        "Insect detection and classification "
        "using a binary detection gate "
        "and feedforward ANN models"
    ),
    version="1.1.0"
)

# -----------------------------------
# 2. ALLOW OUR REACT FRONTEND
# -----------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"]
)

# -----------------------------------
# 3. HOME ENDPOINT
# -----------------------------------

@app.get("/")
def home():

    return {
        "message": "Welcome to InsectClass AI API",
        "status": "running",
        "models": ["single_layer", "multilayer"],
        "insect_detector": "v3",
        "total_classes": len(class_names)
    }

# -----------------------------------
# 4. INSECT CLASSES ENDPOINT
# -----------------------------------

@app.get("/classes")
def get_classes():

    return {
        "total": len(class_names),
        "classes": class_names
    }

# -----------------------------------
# 5. IMAGE PREDICTION ENDPOINT
# -----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    if file.content_type not in [
        "image/jpeg",
        "image/png",
        "image/webp"
    ]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG, PNG, or WebP image."
        )

    # Limit uploads to 10 MB
    contents = await file.read(10 * 1024 * 1024 + 1)

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image exceeds the 10 MB upload limit."
        )

    # Validate and decode the image
    try:
        with Image.open(BytesIO(contents)) as image:
            image.load()
            image = image.convert("RGB")

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="Invalid or unreadable image."
        )

    # Run detector first, then ANNs if appropriate
    results = analyze_image(image)

    return {
        "filename": file.filename,
        **results
    }
