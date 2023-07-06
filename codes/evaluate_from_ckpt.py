import transformers
import argparse
import os
import numpy as np
import re
from char_level_rep import char_level_representation_encoding


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint',
                    required=True,
                    help='path to model checkpoint')
parser.add_argument('--evaluation_set',
                    required=True,
                    help='path to evaluation set as json')
parser.add_argument('--output',
                    required=True,
                    help='file name to save results')
parser.add_argument('--prompt',
                    default = 'None',
                    help='prompt used for the input')
parser.add_argument('--digit',
                    type=eval,
                    choices = [True, False],
                    default='True',
                    help='whether you want digit tokenisation or not')
args = parser.parse_args()

model_checkpoint = args.model_checkpoint
num_epochs = 0 # need large epochs
batch_size = 32
lr = 5e-5 #^-2 too big, -6 too small
prompt = str(args.prompt)

from datasets import load_dataset, load_metric
metric = load_metric("accuracy")

if os.path.exists(args.output) == False:
  os.mkdir(args.output)

category_2_idx = {}
data_files = {"train":'/users/acp21jas/arithmetics/data/train.json'}
for i, category_json in enumerate(os.listdir(args.evaluation_set)):
    data_files[f"eval_{i}"] = f'{args.evaluation_set}/{category_json}' #try name instead of i but creates an error
    name = category_json.strip('.test')
    category_2_idx[i]=name
print(data_files)
raw_datasets = load_dataset('json', data_files=data_files)
max_input_length = 512
max_target_length = 16

from transformers import AutoTokenizer, T5Tokenizer
if 't5' in model_checkpoint:
  tokenizer = T5Tokenizer.from_pretrained(model_checkpoint, model_max_length=max_input_length)
else:
  tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, model_max_length=max_input_length)

num_added_toks = tokenizer.add_tokens(['[F]'], special_tokens=True)

def digit_preprocess_function(examples):
    inputs_ids, attention_mask, questions, answers, equations, labels = [], [], [], [], [], []
    for question, answer, equation in zip(examples['question'], examples['answer'], examples['equation']):
      answer = str(answer)
      inputs = char_level_representation_encoding(question, tokenizer=tokenizer, max_length=max_input_length)
      target = char_level_representation_encoding(answer, tokenizer=tokenizer, max_length=max_target_length)
      inputs_ids.append(inputs['input_ids'])
      attention_mask.append(inputs['attention_mask'])
      questions.append(question)
      answers.append(answer)
      equations.append(re.sub('(\d+\.\d+|\d+)', '#', equation))
      labels.append(target['input_ids'])
    
    model_inputs = {'input_ids':inputs_ids,
                    'attention_mask':attention_mask,
                    'question':questions,
                    'answer':answers,
                    'equation':equations,
                    'labels': labels } 

    return model_inputs

def preprocess_function(examples):
    if prompt == 'Trivia_FLAN':
      inputs = ['Please answer this question: '+doc for doc in examples["question"]] #Trivia_prompt FLAN
    elif prompt == 'webQA':
      inputs = ['Question: '+doc+'Answer: ' for doc in examples["question"]] #Web QA prompt
    elif prompt == 'NT5':
      inputs = [f"answer_me: {doc}" for doc in examples["question"]] # NT5 prompt
    elif prompt == 'Trivia_T0':
      inputs = ["Answer the following question. "+doc for doc in examples["question"]] #Trivia_prompt T0_3B
    elif prompt == 'None':
      inputs = [doc for doc in examples["question"]] # Standard
    else:
      print('ERROR WITH PROMPT')
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True, padding=True) 

    # Setup the tokenizer for targets
    outputs = [doc for doc in examples['answer']]
    labels = tokenizer(outputs, max_length=max_target_length, truncation=True, padding=True)
    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

if args.digit==True:
  print('Tokeniser: digit')
  tokenized_datasets = raw_datasets.map(digit_preprocess_function, batched=True)
else:
  print('Tokeniser: not digit')
  tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)


from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

if 't5' in model_checkpoint:
  model = T5ForConditionalGeneration.from_pretrained(model_checkpoint)
elif 'bloom' in model_checkpoint:
  model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
else:
  model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

model_args = Seq2SeqTrainingArguments(
    output_dir = f'/users/acp21jas/arithmetics/results/evaluation',
    do_predict = True,
    do_train = False,
    do_eval = False,
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.005,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=False,
    push_to_hub=False,
    generation_max_length = max_target_length,
    metric_for_best_model = 'accuracy',
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.999,
    # adam_epsilon = 1e-08,
    # half_precision_backend = "cuda_amp",
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
      pred = pred.replace('\n', '').replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace('<unk>','').replace(' ', '')
      pred_nums = re.findall('\d+\.\d+|\d+', pred)
      if len(pred_nums) > 0:
        pred = pred_nums[-1]
      label = label.replace('\n', '').replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace('<unk>','').replace(' ', '')
      try:
        pred=float(pred) #int(pred) if float(pred)==int(pred) else float(pred) #can just have float(pred) and float(label)
        label=float(label) #int(label) if float(label)==int(label) else float(label)
        # if k in range(160,321+1):
        #   print('predictions:', pred)
        #   print('label:', label)
        #   print()
      except:
        pred = str(pred)
      if str(pred) == str(label):
        acc += 1
        if k in range(0, 153+1):
          op_dict['a+b'] +=1
        elif k in range(154,315+1):
          op_dict['a-b'] +=1
        elif k in range(316,428+1):
          op_dict['a*b'] +=1
        elif k in range(429,530+1):
          op_dict['a/b'] +=1
        elif k in range(531,720+1):
          op_dict['(a+b)-c'] +=1
        elif k in range(721,820+1):
          op_dict['a*(b+c)'] +=1
        elif k in range(821,910+1):
          op_dict['(a+b)/c'] +=1
        elif k in range(911,1010+1):
          op_dict['a*(b-c)'] +=1
        elif k in range(1011,1110+1):
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
    
    op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[154, 162, 113, 102, 190, 100, 90, 100, 100])]
    
    accuracy = accuracy_score(float_preds, float_labels)
    print('MACRO_ACCURACY=', accuracy*100)
    print('OP_ACCURACY=', op_acc)

    # Extract a few results
    result = {}
    result = {key: round(value * 100,4) for key, value in result.items()}
    result['accuracy'] = round(accuracy*100, 4)
    result['op_accuracy'] = op_acc
    return result

trainer = Seq2SeqTrainer(
    model,
    model_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics)

results = {}
for key , value in category_2_idx.items():
    output = trainer.predict(
      test_dataset=tokenized_datasets[f'eval_{key}'],
      max_length=max_target_length)
    print(f'accuracy for {value} = {output[2]}')
    results[value]=[output[2]['test_accuracy']]+ output[2]['test_op_accuracy']

    preds, labels = [], []
    acc = 0
    for i in range(len(output[0])):
        raw_pred = tokenizer.decode(output[0][i])
        pred = raw_pred.replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace('<unk>','').replace(' ', '')
        raw_label = tokenizer.decode(np.where(output[1][i] == -100, 0, output[1][i]))
        label = raw_label.replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace('<unk>','').replace(' ', '')

        try:
          pred = float(pred)
          label = float(label)
        except:
          pred = str(pred)
        preds.append(str(pred))
        labels.append(str(label))

    import csv
    header = ['preds', 'targets']
    data = [ [str(pred), str(label)] for pred, label in zip(preds, labels) ]
    name_value = value.replace(' ', '_')
    with open(f'{args.output}/{name_value}-preds_labels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

import pandas as pd
print(results)
df = pd.DataFrame(data=results)
model_name = model_checkpoint.split('/')[1].split('-')[:2]
df.to_csv(f'{args.output}/summary', index=False, mode='w')
print(model_checkpoint)