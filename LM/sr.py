import time
import speech_recognition as sr
import pyaudio
import logging

logger = logging.getLogger("LMHandler - SpeechRecognition")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log.txt')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

class SpeechRecognition:
    
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.recognizer = sr.Recognizer()

    def getDevices(self):
        devices = self.audio.get_device_count()
        parsed = []
        for i in range(devices):
            device_info = self.audio.get_device_info_by_index(i)
            if (self.audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                parsed.append(f"Microphone: {device_info.get('name')} , Device Index: {device_info.get('index')}")
        return parsed

    def listen(self, device_index=0):

        with self.recognizer.Microphone(device_index=device_index) as source:
            logger.log("Listening!")
            audio = self.recognizer.listen(source)

        start_time = time.time()
        logger.log("Translating from audio...")
        # print(r.recognize_whisper(audio, "tiny", language="english"))
        result = self.recognizer.recognize_vosk(audio)
        logger.log("Translation took", time.time() - start_time, "seconds")
        return result
