import asyncio
import speech_recognition as sr
import pyaudio
from LM import SpeechRecognition
import websockets
import json
import cv2
from io import BytesIO
from PIL import Image

GENERATION_END = "GEN_END"
# sr = SpeechRecognition(device_index=6)

def getVision():
    camera = cv2.VideoCapture(index=0)
    retrieve, frame = camera.read() # OpenCV image is not RGB, but BGR
    camera.release()
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_converted)

    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    return dict(bytes=image_bytes)

async def connect_and_generate():

    async with websockets.connect("ws://192.168.1.249:8000") as websocket:  # Replace with your server URL

        try:
            while True:

                # prompt = sr.listen()
                prompt = input("User: ")
                vision = getVision()

                print(f"Prompt: {prompt}")

                await websocket.send(
                    json.dumps({
                        "type": "generate", 
                        "prompt": prompt, 
                        "image_bytes": vision.get("bytes")
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
