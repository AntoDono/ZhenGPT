from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, GenerationConfig
from typing import List
import numpy as np
import torch
import logging

logger = logging.getLogger("LMHandler")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log.txt')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

class LanguageModel:

    def __init__(self, modelID: str, device="cpu", maxLength: int = 2048, gptq: bool = False, dataset: str = "wikitext", quantize_bits: int = None):

        self.device = device
        self.modelID = modelID
        self.maxLength = maxLength
        self.tokenizer = AutoTokenizer.from_pretrained(modelID)

        self.bos_token = self.tokenizer.bos_token
        self.bos_token_id = self.tokenizer.bos_token_id

        self.eos_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.pad_token = self.tokenizer.pad_token
        self.pad_token_id = self.tokenizer.pad_token_id

        if (gptq):
            gptqConfig = GPTQConfig(
                bits=quantize_bits, 
                tokenizer=self.tokenizer,
                dataset=dataset,
                group_size=128,
            )
            self.model = AutoModelForCausalLM.from_Spretrained(
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

    def generate(self, prompt: str | None, inputs = None, **kwargs):

        if prompt == None and inputs == None:
            Exception(f"Both prompt and inputs are none!")

        skip_special_tokens = kwargs.get("skip_special_tokens") == True
        input_tokens=None

        if inputs == None:
            input_tokens = dict(
                input_ids=self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            )
        else:
            input_tokens = inputs

        if kwargs.get("max_length") == None or kwargs.get("max_length") > self.maxLength:
            logger.warn(f"Requested max length undefined or exceeds maxLength, setting it to {self.maxLength}")
            kwargs["max_length"] = self.maxLength

        kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
        output = self.model.generate(
            **input_tokens,
            generation_config = GenerationConfig(**kwargs)
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
    
    def generateStream(self, prompt: str, stop_token_ids: List[str] = [], stop_keywords: List[str] = [], **kwargs):

        active = True

        def stop():
            nonlocal active
            active = False

        def stream():

            nonlocal active

            skip_special_tokens = kwargs.get("skip_special_tokens") == True
            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            if kwargs.get("max_length") == None or kwargs.get("max_length") > self.maxLength:
                logger.warn(f"Requested max length undefined or exceeds maxLength, setting it to {self.maxLength}")
                kwargs["max_length"] = self.maxLength

            output = None

            with torch.no_grad():  # Disable gradients for efficiency

                for _ in range(kwargs.get("max_length")):
                    
                    attention_mask = torch.ones(len(input_tokens[0])).unsqueeze(0).to(self.device)
                    past_key_values = None
                    
                    if output is not None:
                        past_key_values = output["past_key_values"]

                    ids = self.model.prepare_inputs_for_generation(input_tokens,
                                                            past=past_key_values,
                                                            attention_mask=attention_mask,
                                                            use_cache=True)
                                                
                    output = self.model(**ids)
                    
                    next_token = output.logits[:, -1, :].argmax(dim=-1)

                    input_tokens = torch.cat([input_tokens, torch.tensor([[next_token]]).to(self.device)], dim=-1)

                    token_id = next_token.item() 

                    decoded_output = self.tokenizer.decode(input_tokens[0])
                    stop_early = False

                    for keyword in stop_keywords:
                        if decoded_output[-len(keyword):] == keyword:
                            stop_early = True
                            break
                    
                    if stop_early:
                        break

                    if token_id in stop_token_ids or active == False:
                        break
                    
                    yield self.tokenizer.decode(
                        next_token, skip_special_tokens=skip_special_tokens
                    )

        return stream, stop
                    
                