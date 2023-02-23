from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

########### Used to generate text from a given model ###########

parser = argparse.ArgumentParser()
parser.add_argument("text", help="Text on which the generation will be based")
args = vars(parser.parse_args())

model_path = "./finetuned_model_v4.1"


model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)


inputs = tokenizer(args["text"], return_tensors="pt")
outputs = model.generate(inputs.input_ids, max_length = 200, repetition_penalty = 1.3, attention_mask = inputs.attention_mask, temperature = 2.0)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True)[0])