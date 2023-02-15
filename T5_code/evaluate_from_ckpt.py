import transformers
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-model_checkpoint',
                    required=True,
                    help='path to model checkpoint')
parser.add_argument('-evaluation_set',
                    required=True,
                    help='path to evaluation set as json')
parser.add_argument('-output_name',
                    required=True,
                    help='file name to save results')
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
num_epochs = 0 # need large epochs
batch_size = 4
lr = 2e-5 #^-2 too big, -6 too small
prefix = ""

from datasets import load_dataset, load_metric
metric = load_metric("accuracy")


# categories = ['Original', 'Worded', 'Integers_0_to_1000', '1000+', '1000+_comma_separator', '1000+_space_separator', '1dp_0_to_1000', '2dp_0_to_1000', 'with_initial_0', 'without_initial_0', 'Commuted', 'Random_noise', 'Uniform_noise']

category_2_idx = {}
data_files = {"train":'data/test_train_same/train.json'}
for i, category_json in enumerate(os.listdir(args.evaluation_set)):
  data_files[f'eval_{i}'] = f'{args.evaluation_set}/{category_json}' #try name instead of i but creates an error
  name = category_json.strip('.json')
  category_2_idx[i]=name
raw_datasets = load_dataset('json', data_files=data_files)
max_input_length = 256
max_target_length = 32

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

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model_args = Seq2SeqTrainingArguments(
    output_dir = f'results/evaluation',
    do_predict = True,
    do_train = False,
    do_eval = False,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    generation_max_length = max_target_length,
    metric_for_best_model = 'accuracy',
    adam_beta1 = 0.9,
    adam_beta2 = 0.999,
    adam_epsilon = 1e-08,
    half_precision_backend = "cuda_amp",
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    op_dict = {'a+b':0, 'a-b':0, 'a*b':0, 'a/b':0, '(a+b)-c':0, 'a*(b+c)':0, '(a+b)/c':0, 'a*(b-c)':0, '(a-b)/c':0}
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    acc = 0
    float_preds = []
    float_labels = []
    k = 0
    for pred, label in zip(decoded_preds, decoded_labels):
      pred = pred.replace('\n', '').replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace(' ', '')
      label = label.replace('\n', '').replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace(' ', '')
      try:
        pred=float(pred) #int(pred) if float(pred)==int(pred) else float(pred) #can just have float(pred) and float(label)
        label=float(label) #int(label) if float(label)==int(label) else float(label)
        # if k in range(160,321+1):
        #   print('predictions:', pred)
        #   print('label:', label)
        #   print()
      except:
        pred = 0
        label = 1
      if pred == label:
        acc += 1
        if k in range(0, 159+1):
          op_dict['a+b'] +=1
        elif k in range(160,321+1):
          op_dict['a-b'] +=1
        elif k in range(322,438+1):
          op_dict['a*b'] +=1
        elif k in range(439,562+1):
          op_dict['a/b'] +=1
        elif k in range(563,762+1):
          op_dict['(a+b)-c'] +=1
        elif k in range(763,862+1):
          op_dict['a*(b+c)'] +=1
        elif k in range(863,962+1):
          op_dict['(a+b)/c'] +=1
        elif k in range(963,1062+1):
          op_dict['a*(b-c)'] +=1
        elif k in range(1063,1162+1):
          op_dict['(a-b)/c'] +=1
        else:
          print('problem with operation classification')

      k+=1
      float_preds.append(str(pred))
      float_labels.append(str(label))

    if acc != sum(op_dict.values()):
      print('Error: accuracy not equal to operation accuracy')
      print(acc)
      print(op_dict.values())
      exit()
    
    op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[159, 162, 117, 124, 200, 100, 100, 100, 100])]
    
    accuracy = accuracy_score(float_preds, float_labels)
    print('MACRO_ACCURACY=', accuracy*100)
    print('OP_ACCURACY=', op_acc)

    # Extract a few results
    result = {}
    result = {key: round(value * 100,4) for key, value in result.items()}
    result['test_accuracy'] = round(accuracy*100, 4)
    result['op_accuracy'] = op_acc
    
    return result

trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

results = {}
for key , value in category_2_idx.items():
    output = trainer.predict(
      test_dataset=tokenized_datasets[f'eval_{key}'],
      max_length=max_target_length,
      )
    print(f'accuracy for {value} = {output[2]}')
    results[value]=[output[2]['test_accuracy']]+ output[2]['test_op_accuracy']

import pandas as pd
print(results)
df = pd.DataFrame(data=results)
# model_name = model_checkpoint.split('/')[1].split('-')[:2]
df.to_csv(args.output_name, index=False, mode='a')
# print(model_checkpoint)