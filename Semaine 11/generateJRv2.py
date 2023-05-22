from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM, LlamaTokenizer, BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
print(DEVICE)
tokenizer = LlamaTokenizer.from_pretrained("decapoda-research/llama-7b-hf")
model = LlamaForCausalLM.from_pretrained("decapoda-research/llama-7b-hf", load_in_8bit=True, device_map="auto", quantization_config = quantization_config)

def generate(text, n=1, length_multiplier=3, add_score=False, repetition=1.0, temperature = 1.0):
    input_ids = tokenizer.encode(text, return_tensors='pt').to(DEVICE)
    length = len(input_ids[0])
    beam_outputs = model.generate(input_ids, max_length=length * length_multiplier, top_k=40, temperature=temperature,
                              	do_sample=False,
                              	top_p=0.9, repetition_penalty=repetition, num_return_sequences=n, num_beams=n,
                              	early_stopping=True, return_dict_in_generate=True,  output_scores=True)
    print("Output:\n" + 100 * '-')
    res = []
    for i, beam_output in zip(range(len(beam_outputs.sequences)), beam_outputs.sequences):
        generation = tokenizer.decode(beam_output, skip_special_tokens=True)
        res.append(generation)
        print("{}: {}".format(i, res[-1]))
    return res

