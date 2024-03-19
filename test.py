from TTS.api import TTS

# Initialize the TTS engine
tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)

# Generate speech from text
text = "Hello, this is a test of the TTS library."
# wav = tts.tts(text)

# Save the generated audio to a file
tts.tts_to_file(text, file_path="output.wav")