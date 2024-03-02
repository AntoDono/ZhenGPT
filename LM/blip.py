from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class BLIP():

    def __init__(self, modelID: str, device="cpu", maxLength: int = 2048):
        self.device = device
        self.modelID = modelID
        self.maxLength = maxLength

        self.processor = BlipProcessor.from_pretrained(self.modelID)
        self.model = BlipForConditionalGeneration.from_pretrained(self.modelID).to(self.device)

    def generate(self, prompt: str, raw_image: Image, **kwargs):
        inputs = self.processor(raw_image, prompt, return_tensors="pt", max_length=self.maxLength).to(self.device)
        output = self.model.generate(**inputs)
        return self.processor.decode(output[0], skip_special_tokens=True)