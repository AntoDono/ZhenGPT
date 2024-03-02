from typing import List, Callable
import datetime

class DynamicPrompt:
    
    def __init__(self, enable_history: bool, context: str = "", history: List[str] = [], dynamicContext: Callable = lambda: None, enable_datetime: bool = True):

        self.enable_history = enable_history
        self.context = context
        self.history = history
        self.dynamicContext = dynamicContext
        self.enable_datetime = enable_datetime

    def appendHistory(self, prompt: str):
        if (self.enable_history):
            self.history.append(prompt)
        else:
            raise Exception("History feature is not enabled!")

    def generatePrompt(self, prompt: str, append: bool = False):
        
        processedPrompt = ""

        if self.enable_datetime:
            processedPrompt += "Current datetime: " + str(datetime.datetime.now()) + "\n"

        dynamic = self.dynamicContext()
        if (dynamic == None):
            Exception("Dynamic Context returned None!")
        else:
            processedPrompt += dynamic + "\n"
            
        processedPrompt += self.context + "\n"

        if self.enable_history:
            processedPrompt += "\n".join(self.history) + "\n"
        
        processedPrompt += prompt

        if (append):
            self.history.append(prompt)

        return processedPrompt

