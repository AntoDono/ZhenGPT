import asyncio
import speech_recognition as sr
import pyaudio
from LM import SpeechRecognition
import websockets
import json
import cv2
from LM import BLIP
from PIL import Image

GENERATION_END = "GEN_END"
sr = SpeechRecognition(device_index=6)
blip = BLIP("Salesforce/blip-image-captioning-large", "cpu", maxLength=1024)
camera = cv2.VideoCapture(index=0)

def getVision():
    retrieve, frame = camera.read() # OpenCV image is not RGB, but BGR
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_converted)
    description = blip.generate("I see ", raw_image=image)

    return dict(image=image, description=description.replace("I see ",""))

async def connect_and_generate():

    async with websockets.connect("ws://192.168.1.249:8000") as websocket:  # Replace with your server URL

        try:
            while True:

                prompt = sr.listen()

                print(f"You said: {prompt}")

                await websocket.send(
                    json.dumps({
                        "type": "generate", 
                        "prompt": prompt, 
                        "image_caption": getVision().get("description")
                    })
                )

                print("RESPONSE: ")

                async for word in websocket:
                    if (GENERATION_END == word):
                        break
                    print(word, end="", flush=True)  
                print()
        except KeyboardInterrupt:
            await websocket.send("close")
            await websocket.close()

# Example usage
asyncio.run(connect_and_generate())
