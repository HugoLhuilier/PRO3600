import transformers
from transformers import LlamaTokenizer, LlamaForCausalLM
 
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
)
 
import torch
from datasets import load_dataset
 
import seaborn as sns

#matplotlib inline
sns.set(rc={'figure.figsize':(10, 7)})
sns.set(rc={'figure.dpi':100})
sns.set(style='white', palette='muted', font_scale=1.2)
 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Running on :", DEVICE)

############################################################################
db_path = "..\\..\\Autres ressources PRO3600\\danddbeyond.db"
data_path = "../Data/Semaine_11/danddbeyondDataset.json"
BASE_MODEL = "decapoda-research/llama-7b-hf"
CUTOFF_LEN = 2048
TEST_SIZE = 1000

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT= 0.05
LORA_TARGET_MODULES = [
    "q_proj",
    "v_proj",
]
 
BATCH_SIZE = 4
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = BATCH_SIZE // MICRO_BATCH_SIZE
LEARNING_RATE = 1e-4
TRAIN_STEPS = 3000
OUTPUT_DIR = "../Models/finetuned_model_VL2.0"
############################################################################

"""
db = pickledb.load(db_path, False)
dataset = [json.loads(db.get(e)) for e in db.getall()]
nDataset = [
    {
     "title": e['title'],
     "username": e['comments'][i]['username'],
     "index": e['comments'][i]["index"],
     "comment": e['comments'][i]["comment_text"],
     "next_comment": {
                          "username": e['comments'][i+1]['username'],
                          "comment": e['comments'][i+1]["comment_text"],
                      } if i+1 < len(e['comments']) else None
     }
    for e in dataset
    for i in range(0, len(e['comments']))
    ]
""" #Creation of the dataset

model = LlamaForCausalLM.from_pretrained(
    BASE_MODEL,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)

tokenizer = LlamaTokenizer.from_pretrained(BASE_MODEL)

tokenizer.pad_token_id = (
    0  # unk. we want this to be different from the eos token
)

data = load_dataset("json", data_files=data_path)


def generate_prompt(data_point):
    begin = "[Beginning]" if data_point['index'] == 1 else ""
    title = "[Title: "+ data_point['title']+"]"
    nextCom = "" if data_point["next_comment"] == None else f'{data_point["next_comment"]["username"]}: {data_point["next_comment"]["comment"]}'
    return f"""{title} {begin}

{data_point["username"]}: {data_point["comment"]}

{nextCom}"""

def tokenize(prompt, add_eos_token=True):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=CUTOFF_LEN,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < CUTOFF_LEN
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)
 
    result["labels"] = result["input_ids"].copy()
 
    return result
 
def generate_and_tokenize_prompt(data_point):
    full_prompt = generate_prompt(data_point)
    tokenized_full_prompt = tokenize(full_prompt)
    return tokenized_full_prompt


train_val = data["train"].train_test_split(
    test_size=TEST_SIZE, shuffle=True, seed=42
)
train_data = (
    train_val["train"].map(generate_and_tokenize_prompt)
)
val_data = (
    train_val["test"].map(generate_and_tokenize_prompt)
)


model = prepare_model_for_int8_training(model)
config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGET_MODULES,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()


training_arguments = transformers.TrainingArguments(
    per_device_train_batch_size=MICRO_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=100,
    max_steps=TRAIN_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=10,
    optim="adamw_torch",
    evaluation_strategy="steps",
    save_strategy="steps",
    eval_steps=50,
    save_steps=50,
    output_dir=OUTPUT_DIR,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="tensorboard"
)

data_collator = transformers.DataCollatorForLanguageModeling(
    tokenizer, pad_to_multiple_of=8, return_tensors="pt"
)

trainer = transformers.Trainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=val_data,
    args=training_arguments,
    data_collator=data_collator
)
model.config.use_cache = False
old_state_dict = model.state_dict
model.state_dict = (
    lambda self, *_, **__: get_peft_model_state_dict(
        self, old_state_dict()
    )
).__get__(model, type(model))
 
model = torch.compile(model)
 
trainer.train()
model.save_pretrained(OUTPUT_DIR)