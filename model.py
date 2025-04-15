import os
import glob
from ultralytics import YOLO
import cv2
import numpy as np
from matplotlib import pyplot as plt


CLASS_COLORS = {
    'building': (255, 0, 0),
    'flooded_area': (0, 0, 255),
    'garbage': (0, 255, 0),
    'vehicle': (255, 255, 0)
}
TRAIN_CONFIG = {
    'data': 'data.yaml',
    'epochs': 50,
    'imgsz': 640,
    'batch': 8,
    'name': 'aerial_detection'
}


def find_latest_model():
    detect_folders = glob.glob(os.path.join("runs", "detect", "aerial_detection*"))
    if not detect_folders:
        return None
    latest_folder = max(detect_folders, key=os.path.getmtime)
    model_path = os.path.join(latest_folder, "weights", "best.pt")
    return model_path if os.path.exists(model_path) else None


def train_model():
    print(" No trained model found. Starting training...")
    model = YOLO("yolov8s.pt")
    results = model.train(**TRAIN_CONFIG)
    return find_latest_model()


def load_model():
    model_path = find_latest_model()
    if model_path:
        print(f"✅ Loaded existing model from: {model_path}")
        return YOLO(model_path)
    return YOLO(train_model())


def evaluate_model(model):
    print("\n📊 Evaluating model performance...")
    try:

        metrics = model.val()

        print("\n" + "=" * 85)
        print(f"{'YOLOv8 Validation Metrics':^85}")
        print("=" * 85)
        print(
            f"{'Class':<15}{'Images':>10}{'Instances':>12}{'Precision':>12}{'Recall':>12}{'mAP50':>12}{'mAP50-95':>12}")
        print("-" * 85)

        ### Print class-wise metrics
        for cls_name in model.names.values():
            p = metrics.results_dict.get(f'{cls_name}_precision', 0)
            r = metrics.results_dict.get(f'{cls_name}_recall', 0)
            map50 = metrics.results_dict.get(f'{cls_name}_mAP50', 0)
            map95 = metrics.results_dict.get(f'{cls_name}_mAP50-95', 0)

            print(f"{cls_name:<15}{metrics.results_dict['images']:>10}"
                  f"{metrics.results_dict['instances']:>12}"
                  f"{p:>12.3f}"
                  f"{r:>12.3f}"
                  f"{map50:>12.3f}"
                  f"{map95:>12.3f}")

        ###### Printing overall metrics
        print("-" * 85)
        print(f"{'ALL':<15}{metrics.results_dict['images']:>10}"
              f"{metrics.results_dict['instances']:>12}"
              f"{metrics.box.precision:>12.3f}"
              f"{metrics.box.recall:>12.3f}"
              f"{metrics.box.map50:>12.3f}"
              f"{metrics.box.map:>12.3f}")
        print("=" * 85)

        return metrics

    except Exception as e:
        print(f"\n⚠️ Evaluation failed: {str(e)}")
        print("Continuing with predictions...")
        return None
def visualize_detections(img_rgb, boxes, class_names):
    """Draw boxes with class-specific colors and smart labels"""
    img_height, img_width = img_rgb.shape[:2]

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        class_name = class_names[cls_id]
        color = CLASS_COLORS.get(class_name, (255, 255, 255))

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color, 2)

        label = f"{class_name} {conf:.2f}"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)

        text_x = min(x1, img_width - text_width - 1)
        text_y = y1 - 10 if y1 - 10 > text_height else y1 + 20

        cv2.rectangle(img_rgb,
                      (text_x, text_y - text_height - 2),
                      (text_x + text_width, text_y + 2),
                      color, -1)

        cv2.putText(img_rgb, label, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)


def predict_and_show(model, image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"❌ Image not found at {image_path}")

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img)

    for r in results:
        visualize_detections(img_rgb, r.boxes, model.names)

        plt.figure(figsize=(12, 8))
        plt.imshow(img_rgb)
        plt.axis('off')
        plt.title("YOLOv8 Detection Results")
        plt.show()

        os.makedirs("predictions", exist_ok=True)
        output_path = os.path.join("predictions", os.path.basename(image_path))
        cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
        print(f"💾 Prediction saved to: {output_path}")


if __name__ == "__main__":
    try:
        model = load_model()
        metrics = evaluate_model(model)

        test_images = [
            "AerialData/test/images/chip_0_0DSC07688_JPG.rf.c5f217dfe818daf4f77b3d5edfd3ab8f.jpg",
        ]

        for img_path in test_images:
            print(f"\n🔍 Processing: {img_path}")
            predict_and_show(model, img_path)

    except Exception as e:
        print(f"\n🔥 Error: {str(e)}")
