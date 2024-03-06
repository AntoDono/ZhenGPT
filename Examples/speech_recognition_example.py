import speech_recognition as sr
import pyaudio

# Create an instance of PyAudio
p = pyaudio.PyAudio()

# Get the number of audio I/O devices
devices = p.get_device_count()

# Iterate through all devices
for i in range(devices):
   # Get the device info
   device_info = p.get_device_info_by_index(i)
   # Check if this device is a microphone (an input device)
   print(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")

print()
print()

DEVICE = 6

r = sr.Recognizer()
with sr.Microphone(device_index=DEVICE) as source:
    r.adjust_for_ambient_noise(source=source)
    print("Listening!")
    audio = r.listen(source)

import time
start_time = time.time()
print("Translating from audio...")
# print(r.recognize_whisper(audio, "tiny", language="english"))
print(r.recognize_vosk(audio))
print("Translation took", time.time() - start_time, "seconds")
