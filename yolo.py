from ultralytics import YOLO

model = YOLO("yolov8m.pt")

results = model.predict(
    source="/Volumes/Seagate/CSCI_B657/data/25131/MP4/25131c_day4_merged.mp4",
    save=True,
    conf=0.25,
    imgsz=640,
    stream=True
)

for r in results:
    print(r.save_dir)   # 👈 shows exact folder