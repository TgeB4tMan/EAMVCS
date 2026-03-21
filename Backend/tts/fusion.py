import torch
import torch.nn as nn
import os

class EmotionFusion(nn.Module):
    def __init__(self, speaker_dim=256, emotion_dim=128):
        super().__init__()
        # Enhanced projection expects 512-dim output
        self.projection = nn.Linear(emotion_dim, 512)  # Updated from 256 to 512
        self.adapter = nn.Linear(512, speaker_dim)  # Add adapter to match speaker_dim

        # Search for projection weights in multiple potential locations
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "..", "emotion_training", "projection.pth"),
            os.path.join(os.path.dirname(__file__), "..", "emotion_training", "projection.pth"),
            "emotion_training/projection.pth",
            "projection.pth"
        ]

        proj_path = None
        for path in possible_paths:
            if os.path.exists(path):
                proj_path = path
                break

        if proj_path:
            try:
                state_dict = torch.load(proj_path, map_location="cpu")
                self.load_state_dict(state_dict)
                print(f"✅ Projection weights loaded successfully from {proj_path}")
            except Exception as e:
                print(f"❌ Error loading projection weights: {e}")
                print("WARNING: Using random weights.")
        else:
            print("WARNING: projection.pth not found in any expected location. Using random weights.")


    def forward(self, speaker_emb, emotion_emb, alpha=1.3):
        # Project emotion to 512-dim space
        projected_emotion = self.projection(emotion_emb)
        # Adapt to speaker dimension (256-dim)
        adapted_emotion = self.adapter(projected_emotion)
        # Fuse with speaker embedding
        fused = speaker_emb + alpha * adapted_emotion
        return fused
