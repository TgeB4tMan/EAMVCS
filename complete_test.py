#!/usr/bin/env python3
"""
Complete test script for F5-TTS + Whisper integration
Tests: transcription, translation, and synthesis
"""

import requests
import json
import os
import time
from pathlib import Path

class F5TTSTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.test_audio_dir = Path("test_audio")
        self.test_audio_dir.mkdir(exist_ok=True)
        
    def test_health_check(self):
        """Test if backend is running"""
        print("🔍 HEALTH CHECK")
        print("=" * 50)
        
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print("✅ Backend is running!")
                print(f"📊 System: {data.get('system', 'Unknown')}")
                print(f"📝 Description: {data.get('description', 'Unknown')}")
                return True
            else:
                print(f"❌ Backend returned status: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to backend: {e}")
            return False
    
    def test_transcription(self, audio_path, ref_lang="English"):
        """Test Whisper transcription"""
        print(f"\n🎤 TRANSCRIPTION TEST")
        print("=" * 50)
        print(f"📁 Audio: {audio_path}")
        print(f"🌐 Ref Language: {ref_lang}")
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            return None
            
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audio': audio_file,
                    'ref_lang': (None, ref_lang)
                }
                
                response = requests.post(f"{self.base_url}/transcribe", files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Transcription successful!")
                    print(f"📝 Transcribed Text: '{result.get('text', 'N/A')}'")
                    print(f"🧠 Detected Language: {result.get('detected_language', 'N/A')}")
                    print(f"⏱️ Duration: {result.get('duration', 'N/A')}s")
                    return result
                else:
                    print(f"❌ Transcription failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Transcription error: {e}")
            return None
    
    def test_translation(self, text, source_lang="auto", target_lang="en"):
        """Test translation service"""
        print(f"\n🌍 TRANSLATION TEST")
        print("=" * 50)
        print(f"📝 Source Text: '{text}'")
        print(f"📤 Source Lang: {source_lang}")
        print(f"📥 Target Lang: {target_lang}")
        
        try:
            data = {
                'text': text,
                'source_lang': source_lang,
                'target_lang': target_lang
            }
            
            response = requests.post(f"{self.base_url}/translate", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print("✅ Translation successful!")
                print(f"📝 Translated Text: '{result.get('translated_text', 'N/A')}'")
                print(f"📤 Source Lang: {result.get('source_lang', 'N/A')}")
                print(f"📥 Target Lang: {result.get('target_lang', 'N/A')}")
                return result
            else:
                    print(f"❌ Translation failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Translation error: {e}")
            return None
    
    def test_synthesis(self, text, audio_path, ref_lang="English", gen_lang="en", alpha=0.3):
        """Test F5-TTS synthesis"""
        print(f"\n🎵 F5-TTS SYNTHESIS TEST")
        print("=" * 50)
        print(f"📝 Generation Text: '{text}'")
        print(f"📁 Reference Audio: {audio_path}")
        print(f"🌐 Ref Language: {ref_lang}")
        print(f"🎯 Gen Language: {gen_lang}")
        print(f"🎛️ Alpha: {alpha}")
        
        if not os.path.exists(audio_path):
            print(f"❌ Audio file not found: {audio_path}")
            print("💡 Please update the audio path below")
            return None
            
        try:
            with open(audio_path, 'rb') as audio_file:
                files = {
                    'audio': audio_file,
                    'text': (None, text),
                    'language': (None, gen_lang),
                    'ref_lang': (None, ref_lang),
                    'alpha': (None, str(alpha))
                }
                
                print("🚀 Sending synthesis request...")
                start_time = time.time()
                
                response = requests.post(f"{self.base_url}/synthesize", files=files)
                
                end_time = time.time()
                duration = end_time - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    print("✅ Synthesis successful!")
                    print(f"⏱️  Processing Time: {duration:.2f}s")
                    print(f"🎵 Generated Audio: {result.get('audio_path', 'N/A')}")
                    print(f"📝 Reference Text: '{result.get('ref_text', 'N/A')}'")
                    print(f"🎯 Voice Similarity: {result.get('voice_similarity', 'N/A')}")
                    print(f"🔧 Method: {result.get('synthesis_method', 'N/A')}")
                    
                    # Download the generated audio
                    if 'audio_path' in result:
                        self.download_audio(result['audio_path'])
                    
                    return result
                else:
                    print(f"❌ Synthesis failed: {response.status_code}")
                    print(f"Response: {response.text}")
                    return None
                    
        except Exception as e:
            print(f"❌ Synthesis error: {e}")
            return None
    
    def download_audio(self, audio_filename):
        """Download generated audio"""
        try:
            audio_url = f"{self.base_url}/audio/{audio_filename}"
            response = requests.get(audio_url)
            
            if response.status_code == 200:
                save_path = f"downloaded_{audio_filename}"
                with open(save_path, 'wb') as f:
                    f.write(response.content)
                print(f"📁 Audio downloaded: {save_path}")
                return save_path
            else:
                print(f"❌ Failed to download audio: {response.status_code}")
                return None
                
        except Exception as e:
            print(f"❌ Download error: {e}")
            return None
    
    def create_test_audio(self, text="Hello this is a test recording", filename="test_reference.wav"):
        """Create a simple test audio file path"""
        print(f"\n📝 TEST AUDIO CREATION")
        print("=" * 50)
        print(f"📝 Text: '{text}'")
        print(f"📁 Filename: {filename}")
        print(f"💡 Place your audio file at: {self.test_audio_dir / filename}")
        print(f"💡 Then update the path in test below")
        return str(self.test_audio_dir / filename)
    
    def run_complete_test(self):
        """Run complete test suite"""
        print("🧪 F5-TTS + WHISPER COMPLETE TEST")
        print("=" * 60)
        
        # Test 1: Health check
        if not self.test_health_check():
            print("❌ Backend not running. Please start with: python -m Backend.app")
            return
        
        # Test 2: Create test audio info
        test_audio_path = self.create_test_audio()
        
        print(f"\n📋 INSTRUCTIONS:")
        print(f"1. Place a reference audio file at: {test_audio_path}")
        print(f"2. Update the path in the test below")
        print(f"3. Choose test options:")
        
        # Test menu
        while True:
            print(f"\n" + "─" * 60)
            print("🧪 TEST MENU:")
            print("1. 🎤 Test transcription only")
            print("2. 🌍 Test translation only") 
            print("3. 🎵 Test synthesis only")
            print("4. 🧪 Complete pipeline (transcribe + synthesize)")
            print("5. 🌐 Complete with translation")
            print("6. 📋 Update test audio path")
            print("7. 🌍 Choose reference language")
            print("0. 🚪 Exit")
            print("─" * 60)
            
            choice = input("Choose test (0-7): ").strip()
            
            if choice == "0":
                print("👋 Goodbye!")
                break
            elif choice == "1":
                self.test_transcription(test_audio_path)
            elif choice == "2":
                text = input("Enter text to translate: ")
                self.test_translation(text)
            elif choice == "3":
                text = input("Enter text to synthesize: ")
                self.test_synthesis(text, test_audio_path)
            elif choice == "4":
                text = input("Enter text to synthesize: ")
                self.test_synthesis(text, test_audio_path)
            elif choice == "5":
                text = input("Enter text to translate & synthesize: ")
                # First translate
                trans_result = self.test_translation(text)
                if trans_result:
                    # Then synthesize with translated text
                    self.test_synthesis(
                        trans_result.get('translated_text', text),
                        test_audio_path
                    )
            elif choice == "6":
                new_path = input("Enter audio file path: ")
                if new_path and os.path.exists(new_path):
                    test_audio_path = new_path
                    print(f"✅ Updated test audio path: {test_audio_path}")
                else:
                    print(f"❌ File not found: {new_path}")
            elif choice == "7":
                # Choose reference language
                print("\n🌍 Available Reference Languages:")
                print("1. English")
                print("2. Malayalam")
                
                lang_choice = input("Choose reference language (1-2): ").strip()
                
                if lang_choice == "1":
                    ref_lang = "English"
                    print("✅ Selected: English")
                elif lang_choice == "2":
                    ref_lang = "Malayalam"
                    print("✅ Selected: Malayalam")
                else:
                    print("❌ Invalid choice. Using English.")
                    ref_lang = "English"
                
                # Now test with selected language
                text = input("Enter text to synthesize: ")
                self.test_synthesis(text, test_audio_path, ref_lang=ref_lang)
            else:
                print("❌ Invalid choice. Please try again.")

def main():
    """Main function"""
    print("🚀 F5-TTS + WHISPER TEST SUITE")
    print("=" * 60)
    print("This script tests your F5-TTS backend with:")
    print("✅ Whisper transcription")
    print("✅ Translation services") 
    print("✅ F5-TTS synthesis")
    print("✅ Voice similarity scoring")
    print("✅ Audio download")
    print("=" * 60)
    
    # Get base URL
    base_url = input("Enter backend URL (default: http://localhost:8000): ").strip()
    if not base_url:
        base_url = "http://localhost:8000"
    
    # Create tester
    tester = F5TTSTester(base_url)
    
    # Run tests
    tester.run_complete_test()

if __name__ == "__main__":
    main()
