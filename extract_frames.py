import cv2
import os
import argparse

def extract_frames(video_path, output_folder, frame_rate=5):
    """
    Extracts frames from a video at a specified frame rate.
    
    Parameters:
        video_path (str): Path to the input video.
        output_folder (str): Folder where extracted frames will be saved.
        frame_rate (int): Extract 1 frame every `frame_rate` frames.
    """
    # Determine if the video is real or fake
    label = "real" if "real" in video_path.lower() else "fake"
    labeled_output_folder = os.path.join(output_folder, label)
    
    os.makedirs(labeled_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    success, frame = cap.read()

    while success:
        if frame_count % frame_rate == 0:
            frame_filename = os.path.join(labeled_output_folder, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
        success, frame = cap.read()
        frame_count += 1

    cap.release()
    print(f"✅ Extracted frames from {video_path} saved in {labeled_output_folder}")

def process_video_folder(input_folder, output_folder, frame_rate=5):
    """
    Process all videos in the input folder and its subdirectories.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Supported video extensions
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(video_extensions):
                video_path = os.path.join(root, file)
                try:
                    extract_frames(video_path, output_folder, frame_rate)
                except Exception as e:
                    print(f"❌ Error processing {video_path}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input', required=True, help='Input folder containing videos')
    parser.add_argument('--output', required=True, help='Output folder for extracted frames')
    parser.add_argument('--frame-rate', type=int, default=5, help='Extract 1 frame every N frames')
    
    args = parser.parse_args()
    
    print(f"Processing videos from: {args.input}")
    print(f"Saving frames to: {args.output}")
    print(f"Frame rate: {args.frame_rate}")
    
    process_video_folder(args.input, args.output, args.frame_rate)
    print("\n✅ All videos processed!")

