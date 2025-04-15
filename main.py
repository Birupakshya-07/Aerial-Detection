import os
import glob
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from ultralytics import YOLO
import cv2
import uuid
import numpy as np
from typing import Dict

app = FastAPI()

# Configuration
CLASS_COLORS = {
    'building': (255, 0, 0),  # Red
    'flooded_area': (0, 0, 255),  # Blue
    'garbage': (0, 255, 0),  # Green
    'vehicle': (255, 255, 0)  # Yellow
}


def get_model():
    """Load the latest trained model"""
    try:
        model_path = glob.glob("runs/detect/aerial_detection*/weights/best.pt")[-1]
        return YOLO(model_path)
    except IndexError:
        raise RuntimeError("No trained model found. Please train first.")


def draw_boxes(image, boxes, class_names):
    """Draw bounding boxes on image with larger labels"""
    img_height, img_width = image.shape[:2]

    font_scale = 1.0
    font_thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))


        cv2.rectangle(image, (x1, y1), (x2, y2), color, 3)


        label = f"{class_name} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)


        text_x = min(x1, img_width - text_width - 1)
        text_y = y1 - 15 if y1 - 15 > text_height else y1 + 25  # Increased padding

        cv2.rectangle(image,
                      (text_x, text_y - text_height - 5),  # Increased padding
                      (text_x + text_width + 5, text_y + 5),  # Increased padding
                      color, -1)


        cv2.putText(image, label, (text_x, text_y),
                    font, font_scale, (0, 0, 0), font_thickness)
model = get_model()


@app.get("/")
async def root():
    return JSONResponse({"message": "Aerial Object Detection API is running"})


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict:
    temp_path = None
    try:
        ######## Saving uploaded file temporarily
        file_ext = os.path.splitext(file.filename)[1]
        temp_path = f"temp_{uuid.uuid4()}{file_ext}"

        with open(temp_path, "wb") as buffer:
            buffer.write(await file.read())

        ###### Create results directory if not exists
        os.makedirs("results", exist_ok=True)

        ##### Processing image
        img = cv2.imread(temp_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img)
        output_filename = f"result_{uuid.uuid4()}.jpg"
        output_path = f"results/{output_filename}"

        ### Draw boxes manually
        for r in results:
            draw_boxes(img_rgb, r.boxes, model.names)
            cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

        return {"result_path": output_filename}  # Return only filename, not full path

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


@app.get("/results/{filename}")
async def get_result(filename: str):
    file_path = f"results/{filename}"
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)