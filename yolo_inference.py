from ultralytics import YOLO

model = YOLO('yolov8x')

result = model.predict('data_sample/sample_video.mp4', save=True)
print(result)
print("boxes:")
for box in result[0].boxes:
    print(box)

