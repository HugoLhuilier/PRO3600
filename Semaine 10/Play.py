from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import sys, os

modelPath = "../../Resources/v7.0_epoch_1"
length_mult = 10
rep_pen = 10.


model = AutoModelForCausalLM.from_pretrained(modelPath)
tokenizer = AutoTokenizer.from_pretrained(modelPath)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def log(txt):
    print("Debug :", txt)
    
# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def generate(text, n=5, length_multiplier=3, add_score=False, repetition=1.0, temperature = 1.0, beamSearch = False):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    length = len(input_ids[0])
    if beamSearch:
        outputs = model.generate(input_ids, max_length=length * length_multiplier, top_k=40, temperature=temperature,
                                  	do_sample=True,
                                  	top_p=0.9, repetition_penalty=repetition, num_return_sequences=n, num_beams=n,
                                  	early_stopping=True, return_dict_in_generate=True,  output_scores=True)
    else:
        outputs = model.generate(input_ids, max_length=length * length_multiplier, top_k=40, temperature=temperature,
                                  	do_sample=True,
                                  	top_p=0.9, repetition_penalty=repetition,
                                  	return_dict_in_generate=True,  output_scores=True)
    print("Output:\n" + 100 * '-')
    res = []
    for i, output in zip(range(len(outputs.sequences)), outputs.sequences):
        generation = tokenizer.decode(output, skip_special_tokens=True)
        """
        if add_score:
            generation += "\t" + str(score.item())
            """
        res.append(generation)
        print(res[-1])
    return res


###### Start ######

def start():
    req_tags = input("Enter one or multiple tags, separated by comas ',' : ")
    print("")
    tags = req_tags.split(",")
    sTags = str(tags)
    
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