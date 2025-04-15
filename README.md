# Aerial-Detection
https://github.com/user-attachments/assets/45383b59-2610-4695-842b-7b8a4d8d4351

## Aerial Object Detection using YOLOv8

### Problem Definition

This project aims to detect and classify key objects in aerial drone imagery—specifically **buildings**, **flooded areas**, **garbage**, and **vehicles**. The application scope includes disaster response, urban planning, waste monitoring, and environmental assessments.

---

### Data Annotation and Augmentation

The initial dataset consisted of **90 aerial drone images**, each of which was **manually annotated in Roboflow**. Bounding boxes were created for four target classes:

- `building`
- `flooded_area`
- `garbage`
- `vehicle`

Given the small dataset size, **augmentation** techniques were applied using Roboflow to improve generalization. Augmentations included:

- Random horizontal and vertical flips
- Scaling
- Rotation
- Brightness and contrast adjustments

---

### Model Selection: YOLOv8

The **YOLOv8s** model was selected due to its:

- High-speed, real-time detection capabilities
- Pretrained weights for efficient transfer learning
- Excellent support for deployment (API integration and visualization)
- Lightweight nature, suitable for edge devices

The model was trained using the annotated and augmented dataset with a validation split of 20%.

---

### Evaluation Results

The model was evaluated on a held-out validation set of 68 images. Below are the results:

### Evaluation Metrics

| Class         | Images | Instances | Box(P) | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---------------|--------|-----------|--------|--------|---------|--------------|
| all           | 68     | 797       | 0.492  | 0.370  | 0.355   | 0.199        |
| building      | 68     | 314       | 0.481  | 0.430  | 0.334   | 0.171        |
| flooded_area  | 68     | 100       | 0.823  | 0.580  | 0.709   | 0.457        |
| garbage       | 68     | 129       | 0.220  | 0.124  | 0.092   | 0.037        |
| vehicle       | 68     | 254       | 0.443  | 0.348  | 0.286   | 0.131        |



The model achieved strong performance on **flooded_area** and **building**, with mAP@0.5 values of **0.709** and **0.334** respectively. 

Performance on the **garbage class** was notably lower due to the limited number of training examples and high variation in visual appearance. This indicates a need for further data collection and balancing.

---

### Deployment and Features

Key features of the project:

-  YOLOv8 fine-tuned on custom aerial image dataset
-  Augmented training data using Roboflow
-  Bounding box visualization using OpenCV
-  REST API backend built with FastAPI
-  Streamlit-based frontend dashboard
-  Modular and reusable Python code

---

### Limitations and Future Work

- Increase training examples for underrepresented classes like `garbage`
- Add support for batch image processing and video stream input
- Deploy on a cloud platform for public access (e.g., Render, Hugging Face Spaces)
- Consider incorporating confidence threshold controls in the frontend

---

### Ethical Considerations

- The dataset does not contain personally identifiable or sensitive imagery
- Use of drone images is limited to research and academic purposes
- Class definitions were kept neutral to reduce annotation bias

---

### Conclusion

This project demonstrates how a lightweight, accurate object detection model like YOLOv8 can be fine-tuned and deployed for real-world aerial imagery analysis. By combining a well-annotated dataset, modern detection architecture, and deployment tools like FastAPI and Streamlit, the system provides an effective and interactive solution for visual object detection in aerial drone images.
