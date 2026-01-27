import os
from TTS.api import TTS

# This tells the code where to save the model so it 'pops up' in your folder
os.environ["TTS_HOME"] = os.getcwd() 

# Initialize the model (This will trigger a large download)
# We set gpu=False for now just to ensure it works regardless of your drivers
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", gpu=False)

# Generate a simple test file
# Note: YourTTS REQUIRES a speaker_wav to start, even for a basic test
# You can use any short .wav file you have on your computer
tts.tts_to_file(text="Hello world, I am finally working.", 
                speaker_wav="test_sample.wav", 
                language="en", 
                file_path="output.wav")

print("Success! Check your folder for output.wav")