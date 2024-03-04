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

r = sr.Recognizer()
with sr.Microphone(device_index=6) as source:
    print("Say something!")
    audio = r.listen(source, 3)

print(r.recognize_sphinx(audio))