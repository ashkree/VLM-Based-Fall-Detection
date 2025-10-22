import cv2
import os

def video_to_frames(video_path: str, output_dir: str = "frames") -> None:
    """
    Splits a video into frames and saves them as images.

    Args:
        video_path (str): Path to the video file.
        output_dir (str): Directory where extracted frames will be saved.
    """

    # Ensure the video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    print("Processing video...")

    # Loop through frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit when video ends

        # Construct filename and save frame
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        # Optional: print progress
        if frame_count % 50 == 0:
            print(f"Extracted {frame_count} frames...")

    # Release resources
    cap.release()
    print(f"Done! {frame_count} frames saved in '{output_dir}'")

if __name__ == "__main__":
    video_path = input("Enter the path to the video file: ").strip()
    output_dir = input("Enter output folder name (or press Enter for 'frames'): ").strip() or "frames"
    video_to_frames(video_path, output_dir)
