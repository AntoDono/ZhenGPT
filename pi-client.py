import asyncio
import websockets
import sounddevice as sd
import json
import cv2
from pydub import AudioSegment
from pydub.playback import play
from pydub.effects import speedup
from io import BytesIO
import simpleaudio as sa
from PIL import Image
import base64
from LM.sr import SpeechRecognition
from aioconsole import ainput
import numpy as np
from espeak import Espeak
import espeak
from aioconsole import ainput, aprint
from gpiozero import Servo

espeak.init()

CAMERA_INDEX=0
MIC_INDEX=0
SOCKET_URL="ws://192.168.1.249:8000"
MAX_SIZE_BYTES = 100 * 1024 * 1024
GENERATION_END = "GEN_END"
MOUTH_PIN = 26

sr = SpeechRecognition(device_index=MIC_INDEX)

print('\n'.join(sr.getDevices()))
mic_index = int(input("Which mic device? "))
sr.setDevice(device_index=mic_index)
speaker = espeak.Espeak()
speaker.set_voice(gender=2, variant=4) # 2 is females
speaker.rate = 200
servo = Servo(MOUTH_PIN)
servo.min()

def decodeAudioSegement(audio_base64):
    # Decode the Base64 audio
    audio_bytes = base64.b64decode(audio_base64)
    # Convert the bytes to an audio segment
    audio_segment = AudioSegment.from_file(BytesIO(audio_bytes), format="wav")
    audio_segment.export("temp_audio.wav", format="wav")

def adjustSpeechRate(playback_speed):
    audio = AudioSegment.from_file("temp_audio.wav")
    faster_audio = speedup(audio, playback_speed=playback_speed)
    return faster_audio

async def playAudio(audio_segment):
    audio_segment.export("temp_audio.wav", format="wav")

    wave_obj = sa.WaveObject.from_wave_file("temp_audio.wav")
    play_obj = wave_obj.play()

    movingMouth = asyncio.create_task(moveMouth(0.25))
    while play_obj.is_playing():
       await asyncio.sleep(0.1)  # Check every 0.5 seconds
    movingMouth.cancel()
    resetMouth()

def getVision():
    camera = cv2.VideoCapture(index=CAMERA_INDEX)
    retrieve, frame = camera.read()
    camera.release()
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_converted)

    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    buffer = image_bytes.getvalue()

    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return dict(img_base64=img_base64)

def resetMouth():
    servo.min()
    servo.detach()

async def moveMouth(delaySeconds=0.5):
    servo.attach()
    while True:
        servo.max()
        await asyncio.sleep(delaySeconds)
        servo.min()
        await asyncio.sleep(delaySeconds)

async def ping_intervals(websocket, intervalSeconds):
    print(f"Pinging server at {intervalSeconds}s intervals")
    while True:
        await websocket.send(json.dumps({"type": "ping"}))
        await asyncio.sleep(intervalSeconds)

async def user_handler(websocket):
    while True:
        # prompt = await ainput(">")
        audio = await sr.getAudioFile()
        audio_base64 = sr.encryptAudioDataToBase64(audio_data=audio)
        vision = getVision()

        print(f"Sending audio and image to server...")

        await websocket.send(
            json.dumps({
                "type": "generate-pi", 
                "audio_base64": audio_base64, 
                "img_base64": vision.get("img_base64")
            })
        )

        message = ""

        async for res in websocket:
            response = json.loads(res)
            word = response.get("content")
            if (GENERATION_END == word):
                break
            message += word
            print(word, end="", flush=True)  
        print()

        audio_res = json.loads(await websocket.recv())
        tts_audio_base64 = audio_res.get('content')
        decodeAudioSegement(audio_base64=tts_audio_base64)
        audio_segement = adjustSpeechRate(1.2)
        await playAudio(audio_segement)

async def connect_and_generate():
    async with websockets.connect(SOCKET_URL, max_size=MAX_SIZE_BYTES) as websocket:
        ping_task = asyncio.create_task(ping_intervals(websocket, 10))
        handle_user = asyncio.create_task(user_handler(websocket))

        await asyncio.gather(ping_task, handle_user)

asyncio.run(connect_and_generate())
