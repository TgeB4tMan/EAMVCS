#!/usr/bin/env python3
"""
Test the actual API response to see what's being returned
"""

import requests
import json
import os

def test_synthesize_v2_response():
    """Test the /synthesize-v2 endpoint response structure"""
    base_url = "http://localhost:8000"
    
    print("🧪 Testing /synthesize-v2 Response Structure")
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
    
    # Create a simple test audio file path
    test_audio_path = "test_audio/reference.wav"
    
    if not os.path.exists(test_audio_path):
        print(f"❌ Test audio not found at: {test_audio_path}")
        print("💡 Creating a dummy test file for structure testing...")
        os.makedirs("test_audio", exist_ok=True)
        # Create a minimal WAV file header (44 bytes) for testing
        with open(test_audio_path, 'wb') as f:
            f.write(b'RIFF\x24\x08\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x40\x1f\x00\x00\x80\x3e\x00\x00\x02\x00\x10\x00data\x00\x08\x00\x00')
    
    try:
        with open(test_audio_path, 'rb') as audio_file:
            files = {
                'audio': audio_file,
                'text': (None, "Hello world, this is a test"),
                'target_lang': (None, 'hi'),  # Hindi is supported
                'ref_lang': (None, 'en'),
                'alpha': (None, '0.8')
            }
            
            print("🎤 Sending /synthesize-v2 request...")
            response = requests.post(f"{base_url}/synthesize-v2", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ /synthesize-v2 successful!")
                
                print("\n📋 FULL RESPONSE STRUCTURE:")
                print(json.dumps(result, indent=2))
                
                # Check specifically for emotion data
                print("\n🎭 EMOTION DATA CHECK:")
                print(f"   emotion: {result.get('emotion')}")
                print(f"   emotion_confidence: {result.get('emotion_confidence')}")
                print(f"   emotion_probabilities: {result.get('emotion_probabilities')}")
                print(f"   emotion_profile: {result.get('emotion_profile')}")
                
                # Check nested diagnostics structure
                diagnostics = result.get('diagnostics', {})
                emotion_diag = diagnostics.get('emotion', {})
                print(f"\n📊 NESTED DIAGNOSTICS:")
                print(f"   diagnostics.emotion.predicted: {emotion_diag.get('predicted')}")
                print(f"   diagnostics.emotion.confidence: {emotion_diag.get('confidence')}")
                print(f"   diagnostics.emotion.probabilities: {emotion_diag.get('probabilities')}")
                print(f"   diagnostics.emotion.profile: {emotion_diag.get('profile')}")
                
            else:
                print(f"❌ /synthesize-v2 failed: {response.status_code}")
                print(f"Response: {response.text}")
                
    except Exception as e:
        print(f"❌ Test error: {e}")

if __name__ == "__main__":
    test_synthesize_v2_response()
