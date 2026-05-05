import cv2
import os

ROOT = "/Volumes/Seagate/CSCI_B657/data"

subjects = [25131, 25138, 25176, 25190, 25602]
days = [1, 2, 3]

for subject in subjects:
    for day in days:

        video_path = f"{ROOT}/{subject}/MP4/{subject}c_day{day}_merged.mp4"

        if not os.path.exists(video_path):
            print("Video not found:", video_path)
            continue

        output_folder = f"{ROOT}/{subject}/frames/day{day}"
        os.makedirs(output_folder, exist_ok=True)

        cap = cv2.VideoCapture(video_path)

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps / 5)   # convert to 5 FPS

        count = 0
        saved = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            if count % frame_interval == 0:
                cv2.imwrite(f"{output_folder}/frame_{saved:04d}.jpg", frame)
                saved += 1

            count += 1

        cap.release()

        print(f"Finished subject {subject} day {day} | Frames saved:", saved)

print("All videos processed.")