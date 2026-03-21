#!/usr/bin/env python3
"""
Quick test to verify emotion detection is working in /synthesize endpoint
"""

import requests
import json
import os

def test_emotion_in_synthesize():
    """Test if emotion data is returned in synthesis response"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing Emotion Detection in /synthesize Endpoint")
    print("=" * 60)
    
    # Test health first
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            print("✅ Backend is running")
        else:
            print(f"❌ Backend health check failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to backend: {e}")
        return
    
    # Create a simple test audio file path (you'll need to provide one)
    test_audio_path = "test_audio/reference.wav"
    
    if not os.path.exists(test_audio_path):
        print(f"❌ Test audio not found at: {test_audio_path}")
        print("💡 Please place a reference audio file at the above path and run again")
        print("💡 Or update the test_audio_path variable in this script")
        return
    
    try:
        with open(test_audio_path, 'rb') as audio_file:
            files = {
                'audio': audio_file,
                'text': (None, "Hello, this is a test of emotion detection"),
                'language': (None, 'en'),
                'alpha': (None, '0.8')
            }
            
            print("🎤 Sending synthesis request...")
            response = requests.post(f"{base_url}/synthesize", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Synthesis successful!")
                
                # Check for emotion data
                emotion = result.get('emotion')
                confidence = result.get('emotion_confidence')
                probabilities = result.get('emotion_probabilities')
                profile = result.get('emotion_profile')
                
                print("\n🎭 EMOTION DETECTION RESULTS:")
                print(f"   Detected Emotion: {emotion}")
                print(f"   Confidence: {confidence}")
                print(f"   Probabilities: {probabilities}")
                print(f"   VAD Profile: {profile}")
                
                if emotion and emotion != 'unknown':
                    print("\n✅ EMOTION DETECTION IS WORKING!")
                else:
                    print("\n⚠️ Emotion detection returned 'unknown' or missing")
                
                print(f"\n📁 Generated audio: {result.get('audio_path')}")
                print(f"🎯 Voice similarity: {result.get('voice_similarity')}")
                
            else:
                print(f"❌ Synthesis failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_emotion_in_synthesize()
