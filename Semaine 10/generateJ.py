from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

mPath = "../../Resources/v7.0_epoch_1"


DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained(mPath)
model = AutoModelForCausalLM.from_pretrained(mPath).to(DEVICE)

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
        print("{}: {}".format(i, res[-1]))
    return res
