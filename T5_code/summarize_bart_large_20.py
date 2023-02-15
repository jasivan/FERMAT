import transformers

print(transformers.__version__)
model_checkpoint = "facebook/bart-large"
num_epochs = 50 # need large epochs
batch_size = 2
lr = 2e-5 #^-2 too big, -6 too small
prefix = ""

from datasets import load_dataset, load_metric
metric = load_metric("accuracy")

import json
def make_into_json(path):
  ds = open(f'{path}.source', 'r')
  dt = open(f'{path}.target', 'r')
  source = ds.readlines()
  target = dt.readlines()
  json_list = []
  for q, a in zip(source, target):
    eval_json = {'question':q, 'answer':a}
    json_list.append(eval_json)
  ds.close()
  dt.close()
  with open(f'{path}.json', 'w') as j:
    for i in json_list:
      j.write(json.dumps(i)+'\n')
  return None

# route = 'data/test_train_same' 
route = 'data/T5_train_test' 

# make_into_json(f'{route}/train')
# make_into_json(f'{route}/dev')
# make_into_json(f'{route}/test')
# make_into_json(f'{route}/gsm8k')

data_files = {"train":f'{route}/train.json', "validation":f'{route}/dev.json', "eval":f'{route}/test.json'}

raw_datasets = load_dataset('json', data_files=data_files)

max_input_length = 256
max_target_length = 16

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=max_input_length)

def preprocess_function(examples):
    inputs = [prefix + doc for doc in examples["question"]]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
print(preprocess_function(raw_datasets['train'][:2]))

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir = f"results/{model_name}-test-model",
    evaluation_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    generation_max_length = max_target_length,
    metric_for_best_model = 'accuracy',
    load_best_model_at_end = True,
    greater_is_better = True,
    save_strategy = "epoch",
    eval_delay = 5,
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-08,
    half_precision_backend = "cuda_amp",
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# import nltk
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    float_preds = []
    float_labels = []
    for pred, label in zip(decoded_preds, decoded_labels):
      pred = pred.replace(' ', '').strip('\n').replace('<pad>', '').replace('</s>', '').replace('<s>', '')
      label = label.replace(' ', '').strip('\n').replace('<pad>', '').replace('</s>', '').replace('<s>', '')
      try:
        pred = float(pred)
        label = float(label)
      except:
        pred = 0
        label = 1
      float_preds.append(pred)
      float_labels.append(label)

    result = metric.compute(predictions=float_preds, references=float_labels)#, use_stemmer=True)
    # Extract a few results
    result = {key: value * 100 for key, value in result.items()}
    
    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)
# import nltk
# nltk.download('punkt')
trainer.train()
output = trainer.predict(
    test_dataset=tokenized_datasets['eval'],
    max_length=max_target_length,
)
print('RESULTS:', output[2])

preds = []
labels = []
acc = 0
for i in range(len(output[0])):
  pred = tokenizer.decode(output[0][i]).replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace(' ', '')
  label = tokenizer.decode(np.where(output[1][i] == -100, 0, output[1][i])).replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace(' ', '')
  try:
    pred = float(pred)
    label = float(label)
  except:
    pred = -1
    label = -2
  preds.append(float(pred))
  labels.append(float(label))

result = metric.compute(predictions=preds, references=labels)
print(result)