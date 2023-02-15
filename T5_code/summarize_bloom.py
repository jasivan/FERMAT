import transformers

print(transformers.__version__)
model_checkpoint = "bigscience/bloom-560m"
num_epochs = 20
prefix = ''
# prefix = "summarize: "

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

route = 'data/T5_train_test' 
# make_into_json(f'{route}/train')
# make_into_json(f'{route}/dev')
# make_into_json(f'{route}/test')
# make_into_json(f'{route}/gsm8k')

data_files = {"train":f'{route}/gsm8k.json', "validation":f'{route}/dev.json', "eval":f'{route}/test.json'}

raw_datasets = load_dataset('json', data_files=data_files)

# from transformers import AutoTokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

from transformers import BloomTokenizerFast
tokenizer = BloomTokenizerFast.from_pretrained(model_checkpoint)

max_input_length = 256
max_target_length = 16

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

from transformers import BloomForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
model = BloomForCausalLM.from_pretrained(model_checkpoint)



batch_size = 32
model_name = model_checkpoint.split("/")[-1]
args = Seq2SeqTrainingArguments(
    output_dir = f"results/{model_name}-finetuned",
    evaluation_strategy = "epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    # auto_find_batch_size=True,
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False,
    generation_max_length = max_target_length,
    do_train=True,
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
    
    # Rouge expects a newline after each sentence
    # decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    # decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    
    float_preds = []
    float_labels = []
    for pred, label in zip(decoded_preds, decoded_labels):
      pred = pred.replace(' ', '').strip('\n')
      label = label.replace(' ', '').strip('\n')
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
  pred = tokenizer.decode(output[0][i]).replace('<pad>', '').replace('</s>', '').replace(' ', '')
  label = tokenizer.decode(np.where(output[1][i] == -100, 0, output[1][i])).replace('<pad>', '').replace('</s>', '').replace(' ', '')
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