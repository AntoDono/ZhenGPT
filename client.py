import asyncio
import speech_recognition as sr
import pyaudio
from LM import SpeechRecognition
import websockets
import json

GENERATION_END = "GEN_END"
sr = SpeechRecognition(device_index=6)

async def connect_and_generate():

    async with websockets.connect("ws://localhost:8000") as websocket:  # Replace with your server URL

        try:
            while True:

                prompt = sr.listen()

                print(f"You said: {prompt}")

                await websocket.send(json.dumps({"type": "generate", "prompt": prompt}))

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
