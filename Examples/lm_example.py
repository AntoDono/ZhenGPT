from LM import LanguageModel

model = LanguageModel("microsoft/phi-2", "cuda:0")
history = [
    "User: Hello\n",
    "AI: Hello!\n"
]

while True:
    req = input("User: ")

    history.append(f"User: {req}\nAI:")

    it, stop = model.generateStream(
        '\n'.join(history), 
        stop=[model.tokenizer.convert_tokens_to_ids("\n")],
        num_beams=5,  # Number of beams to explore (e.g., 5)
        no_repeat_ngram_size=2,  # Disallow repeating bigrams (optional)
    )

    print("AI: ", end="")
    res = ""
    for i in it():
        res += i
        print(i, end="", flush = True)

    history[-1] += res
