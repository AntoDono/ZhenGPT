from main import * 
import asyncio
import json
from websockets import serve, WebSocketServerProtocol
import uuid

CONNECTIONS = []
CAPTION = {}
PORT = 8000
GENERATION_END = "GEN_END"

async def handle_connection(websocket: WebSocketServerProtocol):

    global CAPTION, CONNECTIONS

    ID = uuid.uuid4()
    CAPTION[ID] = "None"

    print(f"Client {ID} connected.")
  
    CONNECTIONS.append(websocket.remote_address)
    dynamicPrompt = DynamicPrompt(
        enable_history=True,
        max_length=model.maxLength,
        tokenizer=model.tokenizer,
        context="""
    You are a chatbot called Sammy. Sammy's Vision is a description of what Sammy sees. Speak with the User.""",
        history=[
            f"User: Hello |",
            f"Sammy: Hello, how are you doing? |" 
        ],
        dynamicContext= lambda : f"Sammy's Vision: {CAPTION[ID]}"
    )

    while True:

        data = await websocket.recv()  # Receive data from client

        print(f"Received: {data} from {ID}")

        if data == "close":
            print(f"{ID} closed connection.")
            websocket.close()
            break
        try:
            data_dict = json.loads(data)
            if data_dict["type"] == "generate":
                prompt = data_dict["prompt"]
                for word in generate(user_input=prompt, dynamicPrompt=dynamicPrompt):
                    await websocket.send(word)  # Send response word by word
                await websocket.send(GENERATION_END)

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error processing data: {e}")
            await websocket.send(json.dumps({"error": "Invalid data format"}))

async def start_server():
    async with serve(handle_connection, host="0.0.0.0", port=PORT):
        print(f"Websocket listening at {PORT}")
        await asyncio.Future()  # This line is never reached, but keeps the server running

if __name__ == "__main__":
   asyncio.run(start_server())