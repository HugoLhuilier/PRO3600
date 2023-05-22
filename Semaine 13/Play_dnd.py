from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch
import sys, os
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import torch

base_model = "decapoda-research/llama-7b-hf"
model_str = "../Models/finetuned_model_vL2.0"
length_mult = 10
rep_pen = 10.

tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    model,
    model_str,
    torch_dtype=torch.float16,
)

model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2

model.eval()

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def generate(text, n=1, length_multiplier=3, add_score=False):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    length = len(input_ids[0])
    beam_outputs = model.generate(input_ids = input_ids, max_length=length * length_multiplier, top_k=40, temperature=1.0,
                                  do_sample=False,
                                  top_p=0.9, repetition_penalty=5.0, num_return_sequences=n, num_beams=n,
                                  early_stopping=True, return_dict_in_generate=True,  output_scores=True)
    res = []
    for i, beam_output in zip(range(len(beam_outputs.sequences)), beam_outputs.sequences):
        generation = tokenizer.decode(beam_output, skip_special_tokens=True)
        res.append(generation)
    return res

def log(txt):
    print("Debug :", txt)
    

###### Start ######

def start():
    name = input("Give a name to your story : ")
    print("")
    
    user_input = "Beginning"
    passed_text = sTags + "\n\n" + user_input + "\n\n"
    response = ""
    
    while(True):
        pred = generate(passed_text, n=1, length_multiplier = length_mult, repetition = rep_pen)
        
        log("Prediction done")
        mod_pred = pred[0][len(passed_text):]
        split_pred = mod_pred.split("\n")
        log("Split done")
        
        for e in split_pred:
            if len(e) > 5:
                response = e
                break 
        
        log("Response found")
        
        print(response)
        print("")
        
        user_input = input("Player : ")
        passed_text = sTags + "\n\n" + response + "\n\n" + user_input
