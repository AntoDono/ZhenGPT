from LM import LanguageModel, DynamicPrompt
import random

model = LanguageModel("microsoft/phi-2", "cuda:0")

dp = DynamicPrompt(
    enable_history=True,
    context="You are a helpful chatbot called KE2CMP. Please talk to the user.",
    history=[
        f"User: My name is Jeff |",
        f"KE2CMP: Hello Jeff |" 
    ],
    dynamicContext= lambda : f"Current inflation rate: {random.randint(1, 100)} %"
)

while True:
    req = input("User: ")

    it, stop = model.generateStream(
        dp.generatePrompt(f"User: {req} |", append=True), 
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
