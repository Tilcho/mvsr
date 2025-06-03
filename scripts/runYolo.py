from ultralytics import YOLO

model = YOLO("/home/simon/Documents/MVSR Lab/mvsr/yolo/my_model/my_model.pt", task="OBB")
results=model("/home/simon/Documents/MVSR Lab/mvsr/data/rgb/1.png", show=True, save=True)

result = results[0]  # only one image
with open("detections_0.txt", "w") as f:
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf[0].item()
        cls_id = int(box.cls[0].item())
        f.write(f"{cls_id} {conf:.4f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n")