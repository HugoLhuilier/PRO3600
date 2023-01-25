from transformers import pipeline

pipe = pipeline('text-generation', model='gpt2')

print(pipe("Hello ! What are you ?", num_return_sequences=5))