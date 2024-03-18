import asyncio
import websockets
import json
import cv2
from io import BytesIO
from PIL import Image
import base64
from LM.sr import SpeechRecognition
from aioconsole import ainput
import espeak

CAMERA_INDEX=0
MIC_INDEX=0
SOCKET_URL="ws://192.168.1.249:8000"
MAX_SIZE_BYTES = 10 * 1024 * 1024
GENERATION_END = "GEN_END"

sr = SpeechRecognition(device_index=MIC_INDEX)

print('\n'.join(sr.getDevices()))
mic_index = int(input("Which mic device? "))
sr.setDevice(device_index=mic_index)
espeak.init()
speaker = espeak.Espeak()

def speak_female(text):
    speaker.set_voice("en+f3")
    speaker.rate = 300
    speaker.say(text)

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
                speak_female(message)
                break
            message += word
            print(word, end="", flush=True)  
        print()

async def connect_and_generate():
    async with websockets.connect(SOCKET_URL, max_size=MAX_SIZE_BYTES) as websocket:
        ping_task = asyncio.create_task(ping_intervals(websocket, 10))
        handle_user = asyncio.create_task(user_handler(websocket))

        await asyncio.gather(ping_task, handle_user)

asyncio.run(connect_and_generate())
