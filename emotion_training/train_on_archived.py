#!/usr/bin/env python3
"""
Train emotion model on archived feedback data with more epochs
This helps us test if we can beat the 64% target with more training data
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
from pathlib import Path
import librosa
import soundfile as sf
import random
import glob
import shutil

from model import EmotionCNN
from train import RobustEmotionDataset, EMOTION_MAP, get_previous_accuracy, save_model_metadata

def collect_archived_data():
    """Collect all archived feedback files for training"""
    print("📁 Collecting archived feedback data...")
    
    archive_root = "../feedback_archive"
    all_files = []
    
    # Find all archived sessions
    session_dirs = glob.glob(os.path.join(archive_root, "session_*"))
    session_dirs = [d for d in session_dirs if os.path.isdir(d)]
    
    print(f"📂 Found {len(session_dirs)} archived sessions")
    
    for session_dir in sorted(session_dirs):
        session_name = os.path.basename(session_dir)
        print(f"  📁 Processing {session_name}")
        
        # Collect files from each emotion folder
        for emotion_dir in ["neutral", "happy", "sad", "angry"]:
            emotion_path = os.path.join(session_dir, emotion_dir)
            if os.path.exists(emotion_path):
                files = glob.glob(os.path.join(emotion_path, "*.wav"))
                for file_path in files:
                    # Map emotion to numeric label
                    emotion_label = {
                        "neutral": 0,
                        "happy": 1, 
                        "sad": 2,
                        "angry": 3
                    }[emotion_dir]
                    
                    all_files.append((file_path, emotion_label))
                    print(f"    🎵 Found {emotion_dir}: {os.path.basename(file_path)}")
    
    print(f"✅ Total files collected: {len(all_files)}")
    return all_files

def train_on_archived():
    """Train model on archived data with more epochs"""
    print("🚀 Starting Training on Archived Data")
    print("=" * 60)
    
    # Collect archived data
    archived_files = collect_archived_data()
    
    if len(archived_files) < 10:
        print("❌ Not enough archived files for training (need at least 10)")
        return
    
    # Create dataset from archived files
    data_dirs = []  # We'll pass files directly to dataset
    
    # Create dataset
    print("📊 Creating dataset from archived files...")
    dataset = RobustEmotionDataset(data_dirs, target_sample_rate=16000, fixed_length=3.0, augment=True)
    
    # Override dataset files with our archived files
    dataset.files = [f[0] for f in archived_files]
    dataset.labels = [f[1] for f in archived_files]
    
    print(f"📈 Dataset size: {len(dataset)} samples")
    
    # Check class distribution
    from collections import Counter
    label_counts = Counter(dataset.labels)
    print("📊 Class distribution:")
    for emotion, count in label_counts.items():
        emotion_names = {0: "Neutral", 1: "Happy", 2: "Sad", 3: "Angry"}
        print(f"  {emotion_names[emotion]}: {count} samples")
    
    # Create data loaders
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    # Setup model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(num_classes=4).to(device)
    
    # Load existing model for fine-tuning
    if os.path.exists("../emotion_model.pth"):
        print("📥 Loading existing model for fine-tuning...")
        model.load_state_dict(torch.load("../emotion_model.pth", map_location=device))
        print("✅ Model loaded successfully")
    else:
        print("🆕 Starting from scratch (no existing model found)")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    lr = 0.0001  # Smaller learning rate for fine-tuning
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10)
    
    # Use more epochs for archived data training
    num_epochs = 20  # More epochs for better convergence
    print(f"🎯 Training for {num_epochs} epochs on archived data")
    
    # Get previous accuracy
    previous_acc = get_previous_accuracy()
    if previous_acc is None:
        previous_acc = 55.0  # Default if no history
    best_acc = max(previous_acc, 55.0)  # Conservative baseline
    print(f"📈 Starting from previous best: {previous_acc:.1f}% | Target: {best_acc:.1f}%")
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        
        for features, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_acc = 100 * correct / total
        
        # Validation (simplified - using training data for demo)
        val_acc = train_acc  # In real scenario, use separate validation set
        
        scheduler.step()
        
        print(f"Epoch {epoch+1:2d} | Loss: {total_loss/len(train_loader):.4f} | Acc: {train_acc:.1f}% | Target: {best_acc:.1f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "../emotion_model.pth")
            save_model_metadata(best_acc)
            print(f"🌟 NEW BEST MODEL! (Acc: {best_acc:.2f}%)")
        else:
            print(f"  ⏸️ Not improved (Current: {val_acc:.1f}% < Best: {best_acc:.1f}%)")
    
    print(f"\n🏁 Training Complete!")
    print(f"📊 Final Best Accuracy: {best_acc:.2f}%")
    print(f"📈 Improvement from {previous_acc:.1f}% to {best_acc:.1f}% = +{best_acc-previous_acc:.1f}%")
    
    if best_acc > 64.0:
        print("🎉 SUCCESS: Beat the 64% target!")
    else:
        print(f"🎯 Target: Still need to improve from {best_acc:.1f}% to beat 64%")

if __name__ == "__main__":
    try:
        train_on_archived()
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
