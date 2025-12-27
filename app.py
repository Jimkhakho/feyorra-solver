from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

# ======================
# MODEL (LAZY LOAD)
# ======================
model = None

def get_model():
    global model
    if model is None:
        model = load_model("feyorra_model.h5")
    return model

# ======================
# IMAGE PROCESS (OPTIMIZED)
# ======================
def smart_slice(img_bytes):
    # decode langsung grayscale (hemat RAM)
    img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        return [], []

    img = cv2.resize(img, None, fx=2, fy=2)

    _, th = cv2.threshold(
        img, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    kernel = np.ones((7, 7), np.uint8)
    th = cv2.dilate(th, kernel, 1)

    cnts, _ = cv2.findContours(
        th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 1800:
            boxes.append((x, y, w, h))

    boxes = sorted(boxes, key=lambda b: b[0])[:5]

    imgs = []
    for x, y, w, h in boxes:
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (50, 50)) / 255.0
        imgs.append(roi.reshape(50, 50, 1))

    return np.array(imgs, dtype=np.float32), boxes

# ======================
# ROUTES
# ======================
@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    img_bytes = await file.read()
    imgs, boxes = smart_slice(img_bytes)

    if len(imgs) == 0:
        return {"status": "fail"}

    model = get_model()
    preds = model.predict(imgs, verbose=0)

    idx = int(np.argmax(preds[:, 1]))

    return {
        "status": "success",
        "index": idx,
        "box": boxes[idx]
    }
