from facenet_pytorch import MTCNN
from PIL import Image
import os
import argparse

# Initialize MTCNN with Improved Settings
mtcnn = MTCNN(keep_all=True, min_face_size=10, thresholds=[0.6, 0.7, 0.7])

def extract_faces(frame_folder, output_folder):
    """
    Detects and extracts faces from frames using MTCNN.
    
    Parameters:
        frame_folder (str): Folder containing extracted frames.
        output_folder (str): Folder where cropped faces will be saved.
    """
    # Assign Labels Based on Folder Name
    label = "real" if "real" in frame_folder.lower() else "fake"
    labeled_output_folder = os.path.join(output_folder, label)
    os.makedirs(labeled_output_folder, exist_ok=True)

    processed_images = 0  # Counter for successful face detections

    for frame in sorted(os.listdir(frame_folder)):
        frame_path = os.path.join(frame_folder, frame)

        try:
            img = Image.open(frame_path).convert("RGB")
        except Exception as e:
            print(f"❌ Error reading {frame_path}: {e}")
            continue

        faces, _ = mtcnn.detect(img)
        
        if faces is None:
            print(f"🔸 No faces detected in {frame}. Skipping.")
            continue

        print(f"🟢 Processing {frame} - Detected {len(faces)} face(s)")

        for i, face in enumerate(faces):
            try:
                x1, y1, x2, y2 = map(int, face)
                face_crop = img.crop((x1, y1, x2, y2)).resize((224, 224))
                face_path = os.path.join(labeled_output_folder, f"{frame[:-4]}_face_{i}_{label}.jpg")

                face_crop.save(face_path)
                processed_images += 1
            except Exception as e:
                print(f"❌ Error processing face {i} in {frame}: {e}")

    print(f"✅ Faces extracted and saved in {labeled_output_folder}")
    print(f"📊 Total Faces Saved: {processed_images}")

def process_all_frames(input_folder, output_folder):
    """
    Process both real and fake frames automatically.
    
    Parameters:
        input_folder (str): Base folder containing real and fake frame folders
        output_folder (str): Base folder where faces will be saved
    """
    # Create output directory
    os.makedirs(output_folder, exist_ok=True)
    
    # Process both real and fake folders
    for category in ['real', 'fake']:
        category_path = os.path.join(input_folder, category)
        if os.path.exists(category_path):
            print(f"\n📁 Processing {category} frames...")
            extract_faces(category_path, output_folder)
        else:
            print(f"❌ {category} folder not found in {input_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract faces from frames')
    parser.add_argument('--input', required=True, help='Input folder containing real and fake frame folders')
    parser.add_argument('--output', required=True, help='Output folder for extracted faces')
    
    args = parser.parse_args()
    
    print(f"Processing frames from: {args.input}")
    print(f"Saving faces to: {args.output}")
    
    process_all_frames(args.input, args.output)
    print("\n✅ All frames processed!")
