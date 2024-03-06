import requests
from LM import BLIP
from PIL import Image

model = BLIP("Salesforce/blip-image-captioning-base", "cuda:0", maxLength=1024)

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

print(model.generate(prompt="This photo consists of ", raw_image=raw_image))