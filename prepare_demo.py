import os
import shutil
import sys
import torch
import io

# Add Backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Backend'))
from Backend.emotion_detector import predict_emotion

def prepare_demo():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    print("🚀 Finding the BEST available examples for your teacher demo...")
    
    base_data = r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive"
    target_dir = r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\teacher_demo"
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.makedirs(target_dir, exist_ok=True)
    
    emotions_to_collect = {
        "01": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry"
    }
    
    # Store candidates as (confidence, path, intensity)
    candidates = {emo: [] for emo in emotions_to_collect.values()}
    
    print("Scanning dataset (this may take a few minutes)...")
    
    file_count = 0
    for root, dirs, files in os.walk(base_data):
        for file in files:
            if file.endswith(".wav"):
                parts = file.split("-")
                if len(parts) >= 7 and parts[2] in emotions_to_collect:
                    emotion_code = parts[2]
                    intensity = "STRONG" if parts[3] == "02" else "normal"
                    target_name = emotions_to_collect[emotion_code]
                    
                    file_path = os.path.join(root, file)
                    try:
                        result = predict_emotion(file_path)
                        pred = result['predicted_emotion']
                        conf = result['confidence']
                        
                        # We only want files where the AI agrees with the label
                        if pred == target_name:
                            candidates[target_name].append((conf, file_path, intensity))
                            file_count += 1
                            if file_count % 20 == 0:
                                print(f"Found {file_count} candidate matches...")
                    except Exception:
                        pass
        if all(len(c) >= 5 for c in candidates.values()): # Optimization: Stop if we have enough candidates
            break

    print("\nFinalizing collection...")
    for emotion, list_of_files in candidates.items():
        # Sort by confidence (highest first)
        list_of_files.sort(key=lambda x: x[0], reverse=True)
        
        # Take top 2
        for i, (conf, path, intensity) in enumerate(list_of_files[:2]):
            new_name = f"example_{emotion}_{i+1}.wav"
            shutil.copy(path, os.path.join(target_dir, new_name))
            print(f"✅ Saved {emotion.upper()} ({intensity}): Conf {conf:.1f}%")

    print("\n" + "="*40)
    print(f"📁 DEMO FOLDER READY: {target_dir}")
    print("="*40)

if __name__ == "__main__":
    prepare_demo()
