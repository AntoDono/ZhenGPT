from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2():

    def __init__(self, modelID: str, device="cpu", maxLength: int = 2048):
        self.device = device
        self.modelID = modelID
        self.maxLength = maxLength

        self.processor = Blip2Processor.from_pretrained(self.modelID)
        self.model = Blip2ForConditionalGeneration.from_pretrained(self.modelID).to(self.device)

    def generate(self, prompt: str, raw_image: Image, **kwargs):
        inputs = self.processor(raw_image, prompt, return_tensors="pt", max_length=self.maxLength).to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)