#!/usr/bin/env python3
"""
Simple test for F5-TTS backend
"""

import requests
import json

def simple_test():
    """Simple test with minimal setup"""
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:8000/")
        print("✅ Backend Health Check:")
        print(json.dumps(response.json(), indent=2))
        print()
    except:
        print("❌ Backend not running. Start with: python -m Backend.app")
        return
    
    # Test synthesis with a dummy file (will fail but shows API works)
    print("🧪 Testing Synthesis API...")
    try:
        # This will fail because we don't have a real audio file
        # but it will test if the API endpoints are working
        response = requests.post(
            "http://localhost:8000/synthesize",
            files={
                'text': (None, 'Hello world'),
                'language': (None, 'en'),
                'ref_lang': (None, 'English'),
                'alpha': (None, '0.3'),
                'audio': ('dummy.wav', b'dummy audio data', 'audio/wav')
            }
        )
        
        result = response.json()
        print("📊 API Response:")
        print(json.dumps(result, indent=2))
        
        if 'error' in result:
            print("✅ API is working (error expected with dummy audio)")
        else:
            print("✅ API working perfectly!")
            
    except Exception as e:
        print(f"❌ API Test failed: {e}")

if __name__ == "__main__":
    simple_test()
