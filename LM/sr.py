import time
import speech_recognition as sr
import pyaudio
import json
import logging

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
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source=source)

    def getDevices(self):
        devices = self.audio.get_device_count()
        parsed = []
        for i in range(devices):
            device_info = self.audio.get_device_info_by_index(i)
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                parsed.append(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")
        return parsed

    def listen(self):

        with self.mic as source:
            print("Listening...")
            audio = self.recognizer.listen(source)

        start_time = time.time()
        print("Translating from audio...")
        # result = self.recognizer.recognize_whisper(audio, "tiny", language="english")
        result = json.loads(self.recognizer.recognize_vosk(audio)).get('text')
        print(f"Translation took {time.time() - start_time} seconds")
        return result
