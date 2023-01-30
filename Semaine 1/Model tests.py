from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

tokenizer = AutoTokenizer.from_pretrained("IRealI/GPT2FinetunesSkyrim", use_auth_token="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")

model = AutoModelForCausalLM.from_pretrained("IRealI/GPT2FinetunesSkyrim", use_auth_token="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")

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("Malyn's enchantments are broken.", max_length=500))