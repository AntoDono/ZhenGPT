import asyncio
import speech_recognition as sr
import websockets
import json
import cv2
from io import BytesIO
from PIL import Image

def getVision():
    camera = cv2.VideoCapture(index=0)
    retrieve, frame = camera.read()  # OpenCV image is not RGB, but BGR
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_converted)
    camera.release()

    return dict(image=image)

vision_data = getVision()
image = vision_data.get("image")

# Create a BytesIO object directly from the image data
image_bytes = BytesIO()
image.save(image_bytes, format="JPEG")  # or any other format you prefer
image_bytes = image_bytes.getvalue()

# Create a new Image object from the BytesIO object
image_file = BytesIO(image_bytes)
image = Image.open(image_file)
image.show()