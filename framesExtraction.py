import os

import cv2


class VideoPreprocessor:
    def __init__(self, video_df):
        self.video_df = video_df
        self.frames_directory = 'extracted_frames'
        os.makedirs(self.frames_directory, exist_ok=True)
        self.frame_data = []

    def extract_frames(self):
        print("🔍 Checking video files...")
        total_extracted = 0

        for index, row in self.video_df.iterrows():
            video_path = f"MultipleFiles/{row['id']}"
            if not os.path.exists(video_path):
                print(f"❌ Directory not found: {video_path}")
                continue

            for filename in os.listdir(video_path):
                # print(f"✅ Found video: {filename}")
                extracted = self._extract_frames_from_video(os.path.join(video_path, filename), row)
                total_extracted += extracted

        print(f"📊 Total frames extracted: {total_extracted}")

    def _extract_frames_from_video(self, video_path, row):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Cannot open video: {video_path}")
            return 0

        frame_count, extracted_count = 0, 0

        gender_dir = os.path.join(self.frames_directory, row['Gender'])
        os.makedirs(gender_dir, exist_ok=True)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % 30 == 0:
                frame_filename = os.path.join(gender_dir, f"video_{row['id']}_frame_{frame_count}.jpg")
                cv2.imwrite(frame_filename, frame)

                self.frame_data.append({"Frame": frame_filename, "Gender": row['Gender']})
                extracted_count += 1

            frame_count += 1

        cap.release()
        return extracted_count

    def save_frame_data_to_csv(self):
        if not self.frame_data:
            print("⚠️ No frames extracted, skipping CSV save.")
            return

        frame_df = pd.DataFrame(self.frame_data)
        frame_df.to_csv('extracted_frames_data.csv', index=False)
        print("📁 Frame data saved to extracted_frames_data.csv")
