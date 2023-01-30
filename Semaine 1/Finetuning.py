#################################################################################
#######################         NE FONCTIONNE PAS         #######################
#################################################################################


from transformers import GPT2LMHeadModel, Trainer, AutoTokenizer, pipeline
from transformers import TrainingArguments

data = open("SkyrimScript.txt")
dataset = data.read()
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
split_data = dataset.split("\n\n")
tokenized_dataset = tokenizer(split_data, padding=True)
training_arg = TrainingArguments(output_dir="test_trainer")

#print(split_data)


trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset,
    args=training_arg,
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

print(pipe("This is a whole new story.", num_return_sequences=5))

trainer.train()

print(pipe("This is a whole new story.", num_return_sequences=5))
