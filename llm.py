import os
import torch
import config
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.prompts.prompts import SimpleInputPrompt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
# from llama_index.core import Settings

os.environ["TRANSFORMERS_CACHE"] = "./cache"
local_llm_path = "./llm_local"

def setup_llm():
    if os.path.exists(local_llm_path):
        return load_local_llm(local_llm_path)
    
    return download_save_llm()  
     

def download_save_llm():
    os.system("huggingface-cli login") # Ensure Hugging Face CLI is logged in
    
    llm = HuggingFaceLLM(
        context_window = config.CONTEXT_WINDOW,
        max_new_tokens = config.MAX_NEW_TOKENS,
        generate_kwargs = {"temperature": 0.0, "do_sample": False},
        system_prompt = config.SYSTEM_PROMPT,
        query_wrapper_prompt = SimpleInputPrompt(config.QUERY_WRAPPER_PROMPT),
        tokenizer_name = config.TOKENIZER_NAME,
        model_name = config.MODEL_NAME,
        device_map = "auto",
        model_kwargs = {"torch_dtype": torch.float16, "load_in_8bit": False},  #you can make this true for quantization if using GPU/CUDA
    )
    
    save_llm(llm, local_llm_path)
    
    return llm

def load_llm():
    # os.system("huggingface-cli login") 
    
    llm = HuggingFaceLLM(
        context_window = config.CONTEXT_WINDOW,
        max_new_tokens = config.MAX_NEW_TOKENS,
        generate_kwargs = {"temperature": 0.0, "do_sample": False},
        system_prompt = config.SYSTEM_PROMPT,
        query_wrapper_prompt = SimpleInputPrompt(config.QUERY_WRAPPER_PROMPT),
        tokenizer_name = config.TOKENIZER_NAME,
        model_name = config.MODEL_NAME,
        device_map = "auto",
        model_kwargs = {"torch_dtype": torch.float16, "load_in_8bit": False},
    )
    
    return llm


def save_llm(llm, path):
    model = AutoModelForCausalLM.from_pretrained(llm.model_name)
    tokenizer = AutoTokenizer.from_pretrained(llm.model_name)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)    

def load_local_llm(path):       # configure for llm saved locally
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path)
    text_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)



# #Testing code
# if __name__ == "__main__":
#     llm = setup_llm()
#     print("llm setup done")
#     save_llm(llm, local_llm_path)
#     print("llm saved")
#     load_local_llm(local_llm_path)
#     print("load complete")







