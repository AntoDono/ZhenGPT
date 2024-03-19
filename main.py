from PIL import Image
import warnings
import cv2
from LM.lm import LanguageModel
from LM.blip import BLIP
from LM.prompt import DynamicPrompt

warnings.filterwarnings("ignore")

# model = LanguageModel("microsoft/phi-2", device="cuda:0", maxLength=1024)
model = LanguageModel("TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", device_map={"":"cuda:1"}, device="cuda:1", maxLength=2048)
blip = BLIP("Salesforce/blip-image-captioning-large", "cpu", maxLength=1024)
camera = cv2.VideoCapture(index=0)

def getVision():
    retrieve, frame = camera.read() # OpenCV image is not RGB, but BGR
    rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb_converted)
    description = blip.generate("I see ", raw_image=image)
    
    return dict(image=image, description=description.replace("I see ",""))

def generate(user_input: str, dynamicPrompt: DynamicPrompt):

    prompt = dynamicPrompt.generatePrompt(f"User: {user_input} |", append=True)

    it, stop = model.generateStream(
        prompt, 
        stop_keywords=["|"],
        num_beams=5,  # Number of beams to explore (e.g., 5)
        no_repeat_ngram_size=2,  # Disallow repeating bigrams (optional)
        skip_special_tokens=True
    )

    res = ""
    for i in it():
        res += i
        yield i
        
    dynamicPrompt.appendHistory(f"{res} | ")

if __name__ == "__main__":

    dp = DynamicPrompt(
        enable_history=True,
        max_length=model.maxLength,
        tokenizer=model.tokenizer,
        context="""
    You are a chatbot called Sammy. Sammy's Vision is a description of what Sammy sees. Speak with the User.""",
        history=[
            f"User: Hello |",
            f"Sammy: Hello, how are you doing? |" 
        ],
        dynamicContext= lambda : f"Sammy's Vision: {getVision().get('description')}"
    )

    while True:
        user_input = input("User: ")
        for i in generate(user_input=user_input, dynamicPrompt=dp):
            print(i, end="", flush=True)
        print()