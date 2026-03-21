import os
import sys
import torch
import torchaudio
import numpy as np
import time

# Add Backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Backend'))
from Backend.emotion_detector import predict_emotion, _emotion_names

def generate_report():
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("="*60)
    print("      NEUROVOICE: EMOTION MODEL EVALUATION REPORT")
    print("="*60)
    
    # Check if a new model exists
    model_path = "emotion_training/emotion_model.pth"
    if os.path.exists(model_path):
        m_time = time.ctime(os.path.getmtime(model_path))
        print(f"STATUS: New Model Found!")
        print(f"LAST UPDATED: {m_time}")
    else:
        print("STATUS: Using default/production model.")
        
    print("-" * 60)
    
    # Selection of test files from RAVDESS (if available)
    base_data = r"C:\Users\Baevin\Desktop\MiniProject\EAMVCS\Data\archive"
    
    test_cases = []
    
    # Mapping for RAVDESS: 01=neutral, 03=happy, 04=sad, 05=angry
    emotions_to_find = {
        "01": "neutral",
        "03": "happy",
        "04": "sad",
        "05": "angry"
    }
    
    if os.path.exists(base_data):
        count_per_emotion = {k: 0 for k in emotions_to_find}
        for root, dirs, files in os.walk(base_data):
            for file in files:
                if file.endswith(".wav"):
                    parts = file.split("-")
                    if len(parts) >= 3 and parts[2] in emotions_to_find:
                        code = parts[2]
                        if count_per_emotion[code] < 5: # Test 5 samples per emotion
                            test_cases.append((os.path.join(root, file), emotions_to_find[code]))
                            count_per_emotion[code] += 1
                if all(c >= 5 for c in count_per_emotion.values()): break
            if all(c >= 5 for c in count_per_emotion.values()): break

    if not test_cases:
        print("❌ ERROR: No test data found in Data/archive. skipping metrics.")
        return

    print(f"{'FILE':<25} | {'EXPECTED':<10} | {'PREDICTED':<10} | {'CONFIDENCE'}")
    print("-" * 60)
    
    correct = 0
    total = 0
    
    for path, expected in test_cases:
        filename = os.path.basename(path)
        try:
            result = predict_emotion(path)
            pred = result['predicted_emotion']
            conf = result['confidence']
            
            status = "✅" if pred == expected else "❌"
            if pred == expected: correct += 1
            total += 1
            
            print(f"{filename[:23]+'..':<25} | {expected:<10} | {pred:<10} | {conf:>6.1f}% {status}")
        except Exception as e:
            print(f"Error testing {filename}: {e}")

    accuracy = (correct / total) * 100 if total > 0 else 0
    print("-" * 60)
    print(f"TOTAL SAMPLED: {total} files")
    print(f"OVERALL ACCURACY: {accuracy:.1f}%")
    
    if accuracy > 85:
        print("RESULT: PRODUCTION READY ✨")
    elif accuracy > 70:
        print("RESULT: STABLE BUT NEEDS MORE DATA 📈")
    else:
        print("RESULT: UNDERPERFORMING - CHECK TRAINING LOGS ⚠️")
    
    print("=" * 60)


if __name__ == "__main__":
    generate_report()
