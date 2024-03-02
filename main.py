from LM import BLIP, LanguageModel, DynamicPrompt
from PIL import Image
import warnings
import cv2

warnings.filterwarnings("ignore")

def main_loop():
        
    model = LanguageModel("microsoft/phi-2", device="cuda:0", maxLength=512)
    blip = BLIP("Salesforce/blip-image-captioning-large", "cpu", maxLength=1024)

    def getVision():
        capture = cv2.VideoCapture(0)
        retrieve, frame = capture.read() # OpenCV image is not RGB, but BGR
        rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_converted)
        description = blip.generate("a photograph of ", raw_image=image)

        return dict(image=image, description=description.replace("a photograph of ",""))

    dp = DynamicPrompt(
        enable_history=True,
        max_length=model.maxLength,
        tokenizer=model.tokenizer,
        context="You are a helpful chatbot called KE2CMP. Speak with the user.",
        history=[
            f"User: My name is Jeff |",
            f"KE2CMP: Hello Jeff |" 
        ],
        dynamicContext= lambda : f"You can see: {getVision().get('description')}"
    )

    while True:
        req = input("User: ")

        prompt = dp.generatePrompt(f"User: {req} |", append=True)

        print(f"""==={prompt}===""")

        it, stop = model.generateStream(
            prompt, 
            stop_keywords=["|"],
            num_beams=5,  # Number of beams to explore (e.g., 5)
            no_repeat_ngram_size=2,  # Disallow repeating bigrams (optional)
            skip_special_tokens=False
        )

        res = ""
        for i in it():
            res += i
            print(i.replace("\n", ""), end="", flush = True)
        print()
        dp.appendHistory(f"{res} | ")

if __name__ == "__main__":
    main_loop()