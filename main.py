from LMHandler import LanguageModel

lm = LanguageModel("microsoft/phi-2", "cuda:0")
print(lm.generate("My name is"))