from transformers import AutoModelForSequenceClassification, Trainer, AutoTokenizer, pipeline

data = open("SkyrimScript.txt")
dataset = data.read()
model = AutoModelForSequenceClassification.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_dataset = tokenizer(dataset)

#print(dataset)

trainer = Trainer(
    model=model,
    train_dataset=tokenized_dataset
)

