from main import * 
import asyncio
import json
from websockets import serve, WebSocketServerProtocol
import uuid
from io import BytesIO
import base64
import os
from LM.sr import SpeechRecognition
from TTS.api import TTS

sr = SpeechRecognition(device_index=None)
CONNECTIONS = []
CAPTION = {}
PORT = 8000
GENERATION_END = "GEN_END"
MAX_SIZE_BYTES = 100 * 1024 * 1024

tts = TTS(model_name="tts_models/en/ljspeech/vits", progress_bar=False, gpu=True)

def captionImageFromBase64(img_base64: str):

    buffer = base64.b64decode(img_base64.encode('utf-8'))
    image_file = BytesIO(buffer)
    image = Image.open(image_file)
    description = blip.generate("I see ", raw_image=image)
    
    return dict(image=image, description=description.replace("I see ",""))

def base64AudioFile(generated_text: str):
    audio_bytes = BytesIO()
    tts.tts_to_file(text=generated_text, file_path=audio_bytes)
    audio_base64 = base64.b64encode(audio_bytes.getvalue()).decode('utf-8')
    return audio_base64

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

                audio_base64= data_dict["audio_base64"]
                img_base64 = data_dict["img_base64"]
                audio_file = sr.descryptBase64(audio_base64)
                prompt = await sr.asyncListenAudioFile(audio_file)
                CAPTION[ID] = captionImageFromBase64(img_base64).get("description")

                print(f"\t+Audio prompt: {prompt}")
                print(f"\t+Image caption: {CAPTION[ID]}")

                generated_text = ""

                for word in generate(user_input=prompt, dynamicPrompt=dynamicPrompt):
                    generated_text += word
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

                await websocket.send(
                    json.dumps(
                        dict(type="audio", content=base64AudioFile(generated_text.replace("Sammy: ")))
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
    async with serve(handle_connection, host="0.0.0.0", port=PORT, max_size=MAX_SIZE_BYTES):
        print(f"Websocket listening at {PORT}")
        await asyncio.Future()  # This line is never reached, but keeps the server running

if __name__ == "__main__":
   asyncio.run(start_server())