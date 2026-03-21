def emotion_to_prosody(emotion, confidence=1.0, alpha=0.3):
    """
    Convert emotion label into high-fidelity TTS control tokens.
    'alpha' acts as a direct multiplier for the emotional shift intensity.
    """
    params = {
        "speed": 1.0,
        "pitch": 1.0,
        "volume": 1.0,
        "emphasis": "moderate",
        "pause_factor": 1.0
    }
    
    # Scale intensity
    multiplier = float(alpha)
    base_intensity = 0.6 + (0.4 * float(confidence))
    shift = base_intensity * multiplier
    
    if emotion == "angry":
        # ========== SCREAMING MODE ==========
        # Speed: Lower = faster in YourTTS length scale
        # At max alpha (2.0), shift ≈ 2.0, so speed → 1.0 - 0.6 = 0.4 (very fast)
        params["speed"] = 1.0 - (0.30 * shift)
        # Pitch: Higher = sharper scream
        # At max alpha, pitch → 1.0 + 0.5 = 1.5 (about 7 semitones up = very high)
        params["pitch"] = 1.0 + (0.25 * shift)
        # Volume: LOUD  
        # At max alpha, volume → 1.2 + 0.8 = 2.0 (doubled loudness)
        params["volume"] = 1.2 + (0.40 * shift)
        # No pauses
        params["pause_factor"] = 0.01
        
    elif emotion == "happy":
        params["speed"] = 1.0 - (0.12 * shift)
        params["pitch"] = 1.0 + (0.10 * shift)
        params["volume"] = 1.0
        params["pause_factor"] = 0.6
        
    elif emotion == "sad":
        params["speed"] = 1.0 + (0.50 * shift) 
        params["pitch"] = 1.0 - (0.20 * shift) 
        params["volume"] = 0.8 - (0.15 * shift) 
        params["pause_factor"] = 2.0 
    
    # CLAMP TO STABLE RANGES
    s_val = float(params["speed"])
    p_val = float(params["pitch"])
    v_val = float(params["volume"])
    
    # Speed: 0.35 = nearly 3x speed, 2.5 = very slow
    params["speed"] = max(0.35, min(2.5, s_val))
    # Pitch: 0.70 = deep, 1.6 = very high pitched scream  
    params["pitch"] = max(0.70, min(1.6, p_val))
    # Volume: 0.4 = whisper, 2.5 = extremely loud
    params["volume"] = max(0.4, min(2.5, v_val))
    
    # Debug logging
    print(f"[PROSODY] Emotion={emotion}, Alpha={alpha}, Shift={shift:.2f}")
    print(f"[PROSODY] Speed={params['speed']:.2f}, Pitch={params['pitch']:.2f}, Volume={params['volume']:.2f}")
    
    return params


def apply_emotional_pauses(text, emotion, alpha=0.3):
    """
    Injects punctuation hints for theatrical rhythm based on alpha intensity.
    """
    text = text.strip()
    
    if emotion == "sad":
        frequency = max(1, int(4 / (alpha + 0.1))) 
        words = text.split()
        new_text = []
        for i, word in enumerate(words):
            new_text.append(word)
            if i < len(words) - 1 and (i % frequency == 0):
                new_text.append("...")
        text = " ".join(new_text) + ("..." if alpha > 0.5 else ".")
        
    elif emotion == "angry":
        # SCREAMING: Remove ALL pauses so the AI rushes through everything
        # Strip commas, periods, semicolons - anything that would cause a pause
        for char in ",.;:?!-":
            text = text.replace(char, "")
        # Remove double spaces left behind
        while "  " in text:
            text = text.replace("  ", " ")
        # Add single exclamation at end
        text = text.strip() + "!"
        
    elif emotion == "happy":
        text = text.replace(". ", "! ")
        text = text.replace(", ", " ") 
        if not text.endswith("!"): text += "!"
    
    print(f"[TEXT TRANSFORM] {emotion}: '{text[:80]}...'")
    return text


def post_process_audio(wav_path, pitch_factor, volume_factor):
    """
    High-fidelity post-processing: pitch shift + volume boost + vocal harshness.
    For angry/screaming: adds soft-clip distortion and high-freq emphasis.
    """
    import librosa
    import soundfile as sf
    import numpy as np
    
    print(f"[POST-PROCESS] Starting: pitch={pitch_factor:.2f}, volume={volume_factor:.2f}")
    
    if abs(pitch_factor - 1.0) < 0.01 and abs(volume_factor - 1.0) < 0.01:
        print("[POST-PROCESS] Skipped (no changes needed)")
        return
        
    try:
        y, sr = librosa.load(wav_path, sr=None)
        y = y.astype(np.float64)
        
        # 1. Pitch Shift
        if abs(pitch_factor - 1.0) > 0.01:
            n_steps = 12 * np.log2(pitch_factor)
            print(f"[POST-PROCESS] Pitch shifting by {n_steps:+.2f} semitones")
            y = librosa.effects.pitch_shift(y, sr=sr, n_steps=float(n_steps))
        
        # 2. Volume Amplification
        if volume_factor > 1.0:
            print(f"[POST-PROCESS] Amplifying volume by {volume_factor:.2f}x")
            y = y * volume_factor
        
        # 3. SCREAMING EFFECTS (only when volume > 1.5, i.e. angry with high alpha)
        if volume_factor > 1.3:
            distortion_amount = min(1.0, (volume_factor - 1.3) / 1.2)  # 0.0 to 1.0 scale
            print(f"[POST-PROCESS] Applying SCREAM effects (distortion={distortion_amount:.2f})")
            
            # A) Soft-clip distortion: creates vocal harshness/strain
            # Overdrive the signal then clip it - this is how guitar distortion works
            drive = 1.0 + (4.0 * distortion_amount)  # 1x to 5x overdrive
            y_driven = y * drive
            # Tanh soft-clipping (smooth, musical distortion)
            y = np.tanh(y_driven) * 0.9
            
            # B) High-frequency emphasis (makes voice sound sharper/edgier)
            # Simple first-order high-pass emphasis filter
            emphasis_factor = 0.5 + (0.45 * distortion_amount)  # 0.5 to 0.95
            y_emphasis = np.zeros_like(y)
            y_emphasis[0] = y[0]
            for i in range(1, len(y)):
                y_emphasis[i] = y[i] - emphasis_factor * y[i-1]
            # Mix: blend original with emphasized version
            mix = 0.3 * distortion_amount  # Up to 30% harsh mix
            y = (1.0 - mix) * y + mix * y_emphasis
            
            print(f"[POST-PROCESS] Drive={drive:.1f}x, Emphasis={emphasis_factor:.2f}, Mix={mix:.2f}")
        
        # 4. Soft volume for sad (when volume < 1.0)
        if volume_factor < 1.0:
            print(f"[POST-PROCESS] Softening volume to {volume_factor:.2f}x")
            y = y * volume_factor
            
        # Final clip to prevent digital distortion
        y = np.clip(y, -1.0, 1.0)
        
        sf.write(wav_path, y, sr)
        print(f"[POST-PROCESS] DONE ✅")
        
    except Exception as e:
        print(f"[POST-PROCESS] ERROR: {e}")
        import traceback
        traceback.print_exc()
