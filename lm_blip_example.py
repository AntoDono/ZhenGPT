from LM import BLIP, LanguageModel, DynamicPrompt
from PIL import Image
import warnings
import cv2

warnings.filterwarnings("ignore")

def main_loop():
        
    model = LanguageModel("microsoft/phi-2", device="cuda:0", maxLength=512)
    # model = LanguageModel("TheBloke/CapybaraHermes-2.5-Mistral-7B-GPTQ", device_map="auto", device="cuda:0", maxLength=512)
    blip = BLIP("Salesforce/blip-image-captioning-large", "cpu", maxLength=1024)
    camera = cv2.VideoCapture(0)

    def getVision():
        retrieve, frame = camera.read() # OpenCV image is not RGB, but BGR
        rgb_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_converted)
        description = blip.generate("I see ", raw_image=image)

        return dict(image=image, description=description.replace("I see ",""))

    dp = DynamicPrompt(
        enable_history=True,
        max_length=model.maxLength,
        tokenizer=model.tokenizer,
        context="You are a chatbot called Sammy. Sammy's Vision is a description of what Sammy sees. Speak with the User.",
        history=[
            f"User: Hello |",
            f"Sammy: Hello, how are you doing? |" 
        ],
        dynamicContext= lambda : f"Sammy's Vision: {getVision().get('description')}"
    )

    while True:
        req = input("User: ")

        prompt = dp.generatePrompt(f"User: {req} |", append=True)

        # print(f"""==={prompt}===""")

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
            print(i.replace("\n", ""), end="", flush = True)
        print()
        dp.appendHistory(f"{res} | ")

if __name__ == "__main__":
    main_loop()