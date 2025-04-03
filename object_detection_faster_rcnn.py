# import cv2
# import torch
# import time
# import torchvision.transforms as T
# from torchvision.models.detection import fasterrcnn_resnet50_fpn

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True)
# rcnn_model.to(device)
# rcnn_model.eval()  

# COCO_LABELS = [
#     '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
#     'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe',
#     'eye glasses', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
#     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass',
#     'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
#     'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
#     'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]

# def preprocess_image(image):
#     transform = T.Compose([T.ToTensor()])
#     return transform(image).unsqueeze(0).to(device)

# cap = cv2.VideoCapture(0)

# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     start_time = time.time()
#     ret, frame = cap.read()
#     if not ret:
#         print("Failed to grab frame.")
#         break
    
#     pre_process_start = time.time()
#     image_tensor = preprocess_image(frame)
#     pre_process_time = time.time() - pre_process_start
    
#     inference_start = time.time()
#     with torch.no_grad():
#         outputs = rcnn_model(image_tensor)
#     inference_time = time.time() - inference_start
    
#     post_process_start = time.time()
#     detections_list = []
    
#     for i, (box, score, label) in enumerate(zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels'])):
#         if score > 0.5:  
#             x1, y1, x2, y2 = map(int, box.tolist())
#             class_id = int(label.item())
#             class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
#             confidence = round(float(score.item()), 2)
            
#             detection = {"class": class_name, "confidence": confidence, "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}}
#             detections_list.append(detection)
            
#             cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
#             cv2.putText(frame, f"{class_name} | {confidence:.2f}", (x1, y1 - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
#     post_process_time = time.time() - post_process_start
#     total_time = time.time() - start_time
#     fps = 1.0 / total_time if total_time > 0 else 0.0
    
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), 
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
#     cv2.imshow("Faster R-CNN Live Feed", frame)
#     print(f"Detections: {detections_list}")
#     print(f"Pre-processing Time: {pre_process_time:.4f}s, Inference Time: {inference_time:.4f}s, Post-processing Time: {post_process_time:.4f}s, Total Time: {total_time:.4f}s, FPS: {fps:.2f}")
    
#     if cv2.waitKey(30) == 27:  
#         break

# cap.release()
# cv2.destroyAllWindows()


import cv2
import torch
import time
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Use MPS (Metal Performance Shaders) if available, otherwise fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load Model
rcnn_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
rcnn_model.eval()

# Reduce precision for faster inference
rcnn_model.half()

# COCO Class Labels
COCO_LABELS = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'shoe',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate',
    'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Preprocessing function
def preprocess_image(image):
    transform = T.Compose([T.ToTensor()])
    image_tensor = transform(image).half().unsqueeze(0).to(device)  # Convert to float16
    return image_tensor

# Capture video using AVFoundation (MacOS Optimized)
cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    start_time = time.time()
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to grab frame.")
        break
    
    # Preprocessing
    pre_process_start = time.time()
    image_tensor = preprocess_image(frame)
    pre_process_time = time.time() - pre_process_start
    
    # Inference
    inference_start = time.time()
    with torch.no_grad():
        outputs = rcnn_model(image_tensor)
    inference_time = time.time() - inference_start
    
    # Post-processing & Drawing Bounding Boxes
    post_process_start = time.time()
    detections_list = []
    
    for box, score, label in zip(outputs[0]['boxes'], outputs[0]['scores'], outputs[0]['labels']):
        if score > 0.5:
            x1, y1, x2, y2 = map(int, box.tolist())
            class_id = int(label.item())
            class_name = COCO_LABELS[class_id] if class_id < len(COCO_LABELS) else "Unknown"
            confidence = round(float(score.item()), 2)
            
            detections_list.append({"class": class_name, "confidence": confidence, "bbox": (x1, y1, x2, y2)})
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    post_process_time = time.time() - post_process_start
    total_time = time.time() - start_time
    fps = 1.0 / total_time if total_time > 0 else 0.0

    # Show FPS
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Faster R-CNN Live Feed", frame)
    print(f"FPS: {fps:.2f} | Detections: {detections_list}")
    
    if cv2.waitKey(1) & 0xFF == 27:  # Press 'ESC' to exit
        break

cap.release()
cv2.destroyAllWindows()
