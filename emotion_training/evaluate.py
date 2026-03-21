import os
import sys
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Backend'))
from emotion_detector import predict_emotion

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EMOTION_MAP = {
    '01': 'neutral',
    '02': 'neutral', # Calm mapped to neutral
    '03': 'happy',
    '04': 'sad',
    '05': 'angry'
}

def evaluate_directory(data_dir):
    correct = 0
    total = 0
    files_to_eval = []
    
    folder_map = {
        'neutral': 'neutral',
        'happy': 'happy',
        'sad': 'sad',
        'angry': 'angry',
        'calm': 'neutral'
    }
    
    for root, dirs, files in os.walk(data_dir):
        # Determine emotion from folder name if applicable
        folder_name = os.path.basename(root).lower()
        folder_emotion = folder_map.get(folder_name)
        
        for file in files:
            if file.endswith('.wav'):
                gt_emotion = None
                
                # Check 1: Standard RAVDESS FileName Parsing
                parts = file.split('-')
                if len(parts) >= 3 and parts[2] in EMOTION_MAP:
                    gt_emotion = EMOTION_MAP[parts[2]]
                
                # Check 2: Folder-based label (fallback)
                if not gt_emotion and folder_emotion:
                    gt_emotion = folder_emotion
                    
                if gt_emotion:
                    files_to_eval.append((os.path.join(root, file), gt_emotion))

    if not files_to_eval:
        print(f"Directory: {data_dir} - No applicable files found.")
        return
        
    for file_path, gt_emotion in tqdm(files_to_eval, desc=f"Evaluating {os.path.basename(data_dir)}"):
        try:
            result = predict_emotion(file_path)
            if result['predicted_emotion'] == gt_emotion:
                correct += 1
            total += 1
        except Exception as e:
            print(f"Error evaluating {file_path}: {e}")
            pass
            
    if total > 0:
        accuracy = (correct / total) * 100
        print(f"\n--- Directory: {data_dir} ---")
        print(f"Total Evaluated: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print("-" * 30 + "\n")

if __name__ == "__main__":
    candidate_directories = [
        os.path.join(PROJECT_ROOT, "Data", "archive"),
        os.path.join(PROJECT_ROOT, "training_data"),
    ]
    directories = [path for path in candidate_directories if os.path.exists(path)]
    
    print("Running Comprehensive Evaluation...")
    if not directories:
        print(f"No evaluation directories found. Checked: {candidate_directories}")
    for directory in directories:
        evaluate_directory(directory)
