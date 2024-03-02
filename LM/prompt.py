from typing import List, Callable
from transformers import AutoTokenizer
import datetime

class DynamicPrompt:
    
    def __init__(self, enable_history: bool, max_length: int, tokenizer: AutoTokenizer, context: str = "", history: List[str] = [], dynamicContext: Callable = lambda: None, enable_datetime: bool = True):

        self.enable_history = enable_history
        self.context = context
        self.history = history
        self.dynamicContext = dynamicContext
        self.enable_datetime = enable_datetime
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    def __get_token_length(self, prompt: str):
        return len(self.tokenizer.encode(prompt, return_attention_mask=False))

    def appendHistory(self, prompt: str):
        if (self.enable_history):
            self.history.append(prompt)
        else:
            raise Exception("History feature is not enabled!")

    def generatePrompt(self, prompt: str, append: bool = False):
        
        processedPrompt = ""

        if self.enable_datetime:
            processedPrompt += "Current datetime: " + str(datetime.datetime.now()) + "\n"

        processedPrompt += self.context + "\n"

        dynamic = self.dynamicContext()
        if (dynamic == None):
            Exception("Dynamic Context returned None!")
        else:
            processedPrompt += dynamic + "\n"
            
        if self.enable_history:

            tokens_avaliable = self.max_length - self.__get_token_length(processedPrompt + prompt)
            current_tokens = 0
            historyCombined = ""

            for line in reversed(self.history):

                line_token_length = self.__get_token_length(line)

                if current_tokens + line_token_length > tokens_avaliable:
                    break
                else:
                    current_tokens += line_token_length
                    historyCombined = line + "\n" + historyCombined

            processedPrompt += historyCombined
        
        processedPrompt += prompt + "\n"

        if (append):
            self.history.append(prompt)

        return processedPrompt

