from transformers import GPT2LMHeadModel, Trainer, AutoTokenizer, pipeline

data = open("SkyrimScript.txt")
dataset = data.read()
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_dataset = tokenizer(dataset, max_length=518939)

#print(dataset)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("This is a whole new story.", num_return_sequences=5))

trainer.train()

print(pipe("This is a whole new story.", num_return_sequences=5))