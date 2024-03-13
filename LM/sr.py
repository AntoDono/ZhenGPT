import time
import speech_recognition as sr
import pyaudio
import json
import logging
import asyncio
from io import BytesIO
import base64

logger = logging.getLogger("LMHandler - SpeechRecognition")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('sr-log.txt')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

class SpeechRecognition:
    
    def __init__(self, device_index=None):
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()
        self.deviceIndex = device_index
        self.mic = None

        if self.deviceIndex:
            self.setDevice(self.deviceIndex)

    def setDevice(self, device_index: int):
        self.mic = sr.Microphone(device_index=device_index)
        print("Calibrating Microphone...", end=" ", flush=True)
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source=source, duration=3)
        print("Done")
        
    def getDevices(self):
        devices = self.audio.get_device_count()
        parsed = []
        for i in range(devices):
            device_info = self.audio.get_device_info_by_index(i)
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                parsed.append(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")
        return parsed

    async def listen(self):

        with self.mic as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        start_time = time.time()
        print("Translating from audio...")
        # result = self.recognizer.recognize_whisper(audio, "tiny", language="english")
        result = json.loads(self.recognizer.recognize_vosk(audio)).get('text')
        print(f"Translation took {time.time() - start_time} seconds")
        return result
    
    async def asyncListen(self):
        loop = asyncio.get_running_loop()
        with self.mic as source:
            print("Listening...")
            # Run the blocking operation in a separate thread
            audio = await loop.run_in_executor(None, self.recognizer.listen, source)

        start_time = time.time()
        print("Translating from audio...")
        # Run the blocking operation in a separate thread
        result = await loop.run_in_executor(None, lambda: json.loads(self.recognizer.recognize_vosk(audio)).get('text'))
        # result = await loop.run_in_executor(None, lambda: self.recognizer.recognize_whisper(audio))
        print(f"Translation took {time.time() - start_time} seconds")
        return result
    
    async def asyncListenAudioFile(self, audioFile):
        loop = asyncio.get_running_loop()
        with sr.AudioFile(audioFile) as source:
            print("Listening...")
            # Run the blocking operation in a separate thread
            audio = await loop.run_in_executor(None, self.recognizer.record, source)

        start_time = time.time()
        print("Translating from audio...")
        # Run the blocking operation in a separate thread
        # result = await loop.run_in_executor(None, lambda: json.loads(self.recognizer.recognize_vosk(audio)).get('text'))
        result = await loop.run_in_executor(None, lambda: self.recognizer.recognize_whisper(audio))
        print(f"Translation took {time.time() - start_time} seconds")
        return result
    
    async def getAudioFile(self):
        loop = asyncio.get_running_loop()
        with self.mic as source:
            print("Listening...")
            # Run the blocking operation in a separate thread
            audio = await loop.run_in_executor(None, self.recognizer.listen, source)

        start_time = time.time()
        print("Compiling audio to file...")

        with open("user_audio.wav", "wb") as f:
            f.write(audio.get_wav_data())

        with open("user_audio.wav", "rb") as f:
            audio_data = f.read()
        # result = await loop.run_in_executor(None, lambda: self.recognizer.recognize_whisper(audio))
        print(f"Translation took {time.time() - start_time} seconds")
        return audio_data
    
    def encryptAudioDataToBase64(self, audio_data: bytes):
        audio_bytes = BytesIO(audio_data)
        buffer = audio_bytes.getvalue()

        return base64.b64encode(buffer).decode('utf-8')
    
    def descryptBase64(self, encrypted):
        buffer = base64.b64decode(encrypted.encode('utf-8'))
        audio_data = BytesIO(buffer)
        
        return audio_data
