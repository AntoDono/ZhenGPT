import asyncio
import websockets
import json
import cv2
from io import BytesIO
from PIL import Image
import base64
from LM.sr import SpeechRecognition
from picamera import PiCamera
from picamera.array import PiRGBArray
from aioconsole import ainput

CAMERA_INDEX=0
MIC_INDEX=0
SOCKET_URL="ws://192.168.1.249:8000"
GENERATION_END = "GEN_END"

sr = SpeechRecognition(device_index=MIC_INDEX)

print('\n'.join(sr.getDevices()))
mic_index = int(input("Which mic device? "))
sr.setDevice(device_index=mic_index)


def getVision():
    # Create a PiCamera instance
    camera = PiCamera()
    # Set camera resolution (adjust as needed)
    camera.resolution = (640, 480)
    # Create a numpy array to store the captured frame
    raw_capture = PiRGBArray(camera)
    # Capture a frame from the camera
    camera.capture(raw_capture, format="bgr")
    # Get the numpy array representing the capture
    frame = raw_capture.array
    # Release resources
    camera.close()
    # Convert the frame to RGB format (OpenCV uses BGR by default)
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Convert the NumPy array to a PIL Image
    image = Image.fromarray(rgb_converted)
    
    # Save the image to a BytesIO buffer
    image_bytes = BytesIO()
    image.save(image_bytes, format="JPEG")
    buffer = image_bytes.getvalue()
    
    # Encode the image data as base64
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
        vision = getVision()

        print(f"Sending audio file and image to server...")

        await websocket.send(
            json.dumps({
                "type": "generate-pi", 
                "audio": audio, 
                "img_base64": vision.get("img_base64")
            })
        )

        async for res in websocket:
            response = json.loads(res)
            word = response.get("content")
            if (GENERATION_END == word):
                break
            print(word, end="", flush=True)  
        print()

async def connect_and_generate():
    async with websockets.connect(SOCKET_URL) as websocket:
        ping_task = asyncio.create_task(ping_intervals(websocket, 10))
        handle_user = asyncio.create_task(user_handler(websocket))

        await asyncio.gather(ping_task, handle_user)

asyncio.run(connect_and_generate())
