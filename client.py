import asyncio
import websockets
import json

GENERATION_END = "GEN_END"

async def connect_and_generate():

    async with websockets.connect("ws://localhost:8000") as websocket:  # Replace with your server URL

        try:
            while True:

                prompt = input("Enter your prompt: ")

                await websocket.send(json.dumps({"type": "generate", "prompt": prompt}))

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
