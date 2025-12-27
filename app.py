from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = FastAPI()

model = load_model("feyorra_model.h5")

def smart_slice(img_bytes):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return [], []

    img = cv2.resize(img, None, fx=2, fy=2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((9, 9), np.uint8)
    th = cv2.dilate(th, kernel, 2)

    cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [(x,y,w,h) for (x,y,w,h) in [cv2.boundingRect(c) for c in cnts] if w*h > 2000]
    boxes = sorted(boxes, key=lambda b: b[0])[:5]

    imgs = []
    for x,y,w,h in boxes:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (50,50))/255.0
        imgs.append(roi.reshape(50,50,1))

    return np.array(imgs), boxes

@app.post("/solve")
async def solve(file: UploadFile = File(...)):
    img = await file.read()
    imgs, boxes = smart_slice(img)
    if len(imgs) == 0:
        return {"status":"fail"}

    preds = model.predict(imgs, verbose=0)
    idx = int(np.argmax([p[1] for p in preds]))

    return {"status":"success","index":idx,"box":boxes[idx]}
