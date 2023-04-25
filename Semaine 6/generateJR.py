from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
tokenizer = AutoTokenizer.from_pretrained("./finetuned_model_v6.1/final")
model = AutoModelForCausalLM.from_pretrained("./finetuned_model_v6.1/final").to(DEVICE)

def generate(text, n=5, length_multiplier=3, add_score=False, repetition=1.0, temperature = 1.0):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    length = len(input_ids[0])
    beam_outputs = model.generate(input_ids, max_length=length * length_multiplier, top_k=40, temperature=temperature,
                              	do_sample=True,
                              	top_p=0.9, repetition_penalty=repetition, num_return_sequences=n, num_beams=n, 
                              	early_stopping=True, return_dict_in_generate=True,  output_scores=True)
    print("Output:\n" + 100 * '-')
    res = []
    for i, beam_output, score in zip(range(len(beam_outputs.sequences)), beam_outputs.sequences, beam_outputs.sequences_scores):
        generation = tokenizer.decode(beam_output, skip_special_tokens=True)
        if add_score:
            generation += "\t" + str(score.item())
        res.append(generation)
        print("{}, {}: {}".format(i, score, res[-1]))
    return res
