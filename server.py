from main import * 
import asyncio
import json
from websockets import serve, WebSocketServerProtocol
import uuid
from io import BytesIO
import base64
import os

CONNECTIONS = []
CAPTION = {}
PORT = 8000
GENERATION_END = "GEN_END"

def captionImageFromBase64(img_base64: str):

    buffer = base64.b64decode(img_base64.encode('utf-8'))
    image_file = BytesIO(buffer)
    image = Image.open(image_file)
    description = blip.generate("I see ", raw_image=image)
    
    return dict(image=image, description=description.replace("I see ",""))

async def handle_connection(websocket: WebSocketServerProtocol):

    global CAPTION, CONNECTIONS

    ID = uuid.uuid4()
    CAPTION[ID] = "None"

    print(f"Client {ID} connected.")

    with open(f"{os.getcwd()}/contextPrompt.txt") as f:
        context = f.read()
        f.close()
  
    CONNECTIONS.append(websocket.remote_address)
    dynamicPrompt = DynamicPrompt(
        enable_history=True,
        max_length=model.maxLength,
        tokenizer=model.tokenizer,
        context=context,
        history=[
            f"User: Hello |",
            f"Sammy: Hello, how are you doing? |" 
        ],
        dynamicContext= lambda : f"Sammy's Vision: {CAPTION[ID]}"
    )

    while True:

        data = await websocket.recv()  # Receive data from client

        print(f"Received request from {ID}")

        if data == "close":
            print(f"{ID} closed connection.")
            websocket.close()
            break
        try:

            data_dict = json.loads(data)

            if data_dict["type"] == "generate":
                
                print(f"\t+ Generate Request")

                prompt = data_dict["prompt"]
                img_base64 = data_dict["img_base64"]
                CAPTION[ID] = captionImageFromBase64(img_base64).get("description")

                for word in generate(user_input=prompt, dynamicPrompt=dynamicPrompt):
                    await websocket.send(
                        json.dumps(
                            dict(type="response", content=word)
                        )
                    )  # Send response word by word
                
                await websocket.send(
                    json.dumps(
                        dict(type="response", content=GENERATION_END)
                    )
                )
            elif data_dict["type"] == "generate-pi":
                
                print(f"\t+ Generate-Pi Request")

                audio= data_dict["audio"]
                img_base64 = data_dict["img_base64"]
                audio_file = BytesIO(audio)
                CAPTION[ID] = captionImageFromBase64(img_base64).get("description")

                for word in generate(user_input=prompt, dynamicPrompt=dynamicPrompt):
                    await websocket.send(
                        json.dumps(
                            dict(type="response", content=word)
                        )
                    )  # Send response word by word
                
                await websocket.send(
                    json.dumps(
                        dict(type="response", content=GENERATION_END)
                    )
                )
            elif data_dict["type"] == "ping":
                print(f"\t+ Ping Request")
            else:
                print(f"\tX Unknown Request")

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing data: {e}")
            await websocket.send(json.dumps({"error": "Invalid data format"}))

async def start_server():
    async with serve(handle_connection, host="0.0.0.0", port=PORT):
        print(f"Websocket listening at {PORT}")
        await asyncio.Future()  # This line is never reached, but keeps the server running

if __name__ == "__main__":
   asyncio.run(start_server())