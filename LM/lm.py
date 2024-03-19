from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig, GenerationConfig
from typing import List
import numpy as np
import torch
import logging

logger = logging.getLogger("LMHandler - LM")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('log.txt')
logger.addHandler(file_handler)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

class LanguageModel:

    def __init__(self, modelID: str, device:str ="cpu", device_map: str = None, trust_remote_code: bool = False, maxLength: int = 2048, torch_dtype: torch.dtype = "auto", gptq: bool = False, dataset: str = "wikitext2", quantize_bits: int = None):
        
        self.modelID = modelID
        self.maxLength = maxLength
        self.tokenizer = AutoTokenizer.from_pretrained(modelID, max_length=self.maxLength, truncation=True, use_fast=True)
        self.device = torch.device(device=device)
        self.device_map = device_map
        self.torch_dtype = torch_dtype

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
                trust_remote_code=trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                modelID,
                quantization_config=gptqConfig,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                modelID, 
                load_in_4bit = quantize_bits==4,
                load_in_8bit = quantize_bits==8,
                trust_remote_code=trust_remote_code,
                torch_dtype=self.torch_dtype,
                device_map=self.device_map
            )

        self.model = self.model.to(self.device)

    def generate(self, prompt: str | None, max_new_tokens: int = 100, inputs = None, **kwargs):

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

        kwargs["max_new_tokens"] = max_new_tokens
        kwargs["max_length"] = self.maxLength
        kwargs["bos_token_id"] = self.tokenizer.bos_token_id
        kwargs["eos_token_id"] = self.tokenizer.eos_token_id
        kwargs["pad_token_id"] = self.tokenizer.pad_token_id
            
        output = self.model.generate(
            **input_tokens,
            generation_config = GenerationConfig(**kwargs)
        )

        return self.tokenizer.decode(output[0], skip_special_tokens=skip_special_tokens)
    
    def generateStream(self, prompt: str, max_new_tokens: int = 100, stop_token_ids: List[str] = [], stop_keywords: List[str] = [], **kwargs):

        active = True

        def stop():
            nonlocal active
            active = False

        def stream():

            nonlocal active

            skip_special_tokens = kwargs.get("skip_special_tokens") == True
            input_tokens = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)


            kwargs["max_length"] = self.maxLength
            kwargs["max_new_tokens"] = max_new_tokens

            output = None
            prev_decoded_output = prompt

            for _ in range(max_new_tokens):
                    
                attention_mask = torch.ones(len(input_tokens[0])).unsqueeze(0).to(self.device)
                past_key_values = None
                    
                if output is not None:
                    past_key_values = output["past_key_values"]

                ids = self.model.prepare_inputs_for_generation(input_tokens,
                                                        past=past_key_values,
                                                        attention_mask=attention_mask,
                                                        use_cache=True).to(self.device)
                                                
                output = self.model(**ids)
                    
                next_token = output.logits[:, -1, :].argmax(dim=-1)

                input_tokens = torch.cat([input_tokens, torch.tensor([[next_token]]).to(self.device)], dim=-1)

                token_id = next_token.item() 

                decoded_output = self.tokenizer.decode(input_tokens[0], skip_special_tokens=skip_special_tokens)
                stop_early = False

                for keyword in stop_keywords:
                    if decoded_output[-len(keyword):] == keyword:
                        stop_early = True
                        break
                    
                if stop_early:
                    break

                if token_id in stop_token_ids or active == False:
                    break
                    
                yield decoded_output[len(prev_decoded_output):]

                prev_decoded_output = decoded_output

        return stream, stop
                    
                