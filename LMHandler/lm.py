from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
from typing import List
import torch

class LanguageModel:

    def __init__(self, modelID: str, device="cpu", gptq: bool = False, dataset: str = "wikitext", quantize_bits: int = None):

        self.device = device
        self.modelID = modelID
        self.tokenizer = AutoTokenizer.from_pretrained(modelID)

        if (gptq):
            gptqConfig = GPTQConfig(
                bits=quantize_bits, 
                tokenizer=self.tokenizer,
                dataset=dataset,
                group_size=128,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                modelID,
                quantization_config=gptqConfig,
            ).to(self.device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                modelID, 
                torch_dtype=torch.float16, 
                load_in_4bit = quantize_bits==4,
                load_in_8bit = quantize_bits==8,
            ).to(self.device)

    def generate(self, prompt: str, max_length:int=100):

        input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        output = self.model.generate(
            input_ids=input_tokens,
            max_length=max_length
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=True)