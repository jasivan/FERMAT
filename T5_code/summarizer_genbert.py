import transformers
import argparse
from char_level_rep import char_level_representation_encoding


parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint',
                    help='path to model checkpoint')
parser.add_argument('--route',
                    help='path to train/dev/test set in json format')
parser.add_argument('--train',
                    default='train.json',
                    help='name of training set in json format')
parser.add_argument('--dev',
                    default='dev.json',
                    help='name of development set in json format')
parser.add_argument('--test',
                    default='test.json',
                    help='name of test set in json format')
parser.add_argument('--output',
                    help='path to output dir')
parser.add_argument('--epochs',
                    type=int,
                    default=5,
                    help='number of epochs')
parser.add_argument('--lr',
                    type=float,
                    default=1e-5, #check examples of learning rate
                    help='learning rate')
parser.add_argument('--weight',
                    type=float,
                    default=0.005,
                    help='weight decay')
parser.add_argument('--batch',
                    type=int,
                    default=32,
                    help='batch size')
parser.add_argument('--fp16',
                    type=eval,
                    choices = [True, False],
                    default='True',
                    help='whether you want fp16 or not')
args = parser.parse_args()

print(args)
model_checkpoint = args.model_checkpoint
num_epochs = args.epochs # need large epochs
batch_size = args.batch
lr = args.lr #^-2 too big, -6 too small

from datasets import load_dataset, load_metric
metric = load_metric("accuracy")

import json
route = args.route
train = args.train
dev = args.dev
test = args.test

data_files = {"train":f'{route}/{train}', "validation":f'{route}/{dev}', "eval":f'{route}/{test}'}
print(data_files)
raw_datasets = load_dataset('json', data_files=data_files)

max_input_length = 512 #  truncate to max length
max_target_length = 64 # truncate to max length


def preprocess_function(examples):
    # inputs = [doc for doc in examples["question"]]
    # model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True) 

    # # Setup the tokenizer for targets
    # outputs = [doc for doc in examples['answer']]
    # labels = tokenizer(outputs, max_length=max_target_length, truncation=True)
    # model_inputs["labels"] = labels["input_ids"]

    inputs_ids, attention_mask, questions, answers, labels = [], [], [], [], []
    for question, answer in zip(examples['question'], examples['answer']):
      inputs = char_level_representation_encoding(question, tokenizer=tokenizer, max_length=max_input_length)
      target = char_level_representation_encoding(answer, tokenizer=tokenizer, max_length=max_target_length)
      inputs_ids.append(inputs['input_ids'])
      attention_mask.append(inputs['attention_mask'])
      questions.append(question)
      answers.append(answer)
      labels.append(target['input_ids'])
    
    model_inputs = {'input_ids':inputs_ids,
                    'attention_mask':attention_mask,
                    'question':questions,
                    'answer':answers,
                    'labels': labels } 


    return model_inputs

from transformers import AutoModel, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

# model = AutoModel.from_pretrained(model_checkpoint, is_decoder=True, add_cross_attention = True)
from transformers import EncoderDecoderModel
model = EncoderDecoderModel.from_encoder_decoder_pretrained(model_checkpoint, model_checkpoint)
from transformers import BertTokenizerFast
# Set tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_checkpoint)
num_added_toks = tokenizer.add_tokens(['[F]'], special_tokens=True)
tokenizer.bos_token = tokenizer.cls_token
tokenizer.eos_token = tokenizer.sep_token

# Set model's config
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model_name = model_checkpoint.split("/")[-1]
model_args = Seq2SeqTrainingArguments(
    output_dir = args.output,
    evaluation_strategy = "epoch",
    learning_rate=lr,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=args.weight,
    save_total_limit=1,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    fp16=args.fp16,
    push_to_hub=False,
    generation_max_length = max_target_length,
    generation_num_beams = 5,
    metric_for_best_model = 'accuracy',
    load_best_model_at_end = True,
    greater_is_better = True,
    save_strategy = "epoch",
    warmup_ratio = 0.1,
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.999,
    # adam_epsilon = 1e-08,
    # half_precision_backend = "cuda_amp",
)
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
# import nltk
import numpy as np

from sklearn.metrics import accuracy_score
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    acc = 0
    float_preds = []
    float_labels = []
    for pred, label in zip(decoded_preds, decoded_labels):
      pred = pred.replace('\n', '').replace(' ', '').replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
      label = label.replace('\n', '').replace(' ', '').replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
      # print('prediction:', pred)
      # print('label:', label)
      if pred == label:
        acc += 1
      float_preds.append(str(pred))
      float_labels.append(str(label))

    accuracy = accuracy_score(float_preds, float_labels)
    print('ACCURACY=', accuracy*100)
    print('MANUAL ACCURACY=', acc)

    # Extract a few results
    result = {}
    result['accuracy'] = accuracy
    result['count_accuracy'] = acc/100
    result = {key: value * 100 for key, value in result.items()}
    
    return {k: round(v, 4) for k, v in result.items()}

trainer = Seq2SeqTrainer(
    model=model,
    args=model_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,  
    tokenizer=tokenizer,
    compute_metrics=compute_metrics
)

trainer.train()
output = trainer.predict(
                        test_dataset=tokenized_datasets['eval'],
                        max_length=max_target_length,
                        num_beams = 5
                        )

print('RESULTS:', output[2])

preds = []
labels = []
acc = 0
for i in range(len(output[0])):
  pred = tokenizer.decode(output[0][i]).replace(' ', '').replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')
  label = tokenizer.decode(np.where(output[1][i] == -100, 0, output[1][i])).replace(' ', '').replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '')

  if str(pred) == str(label):
    acc += 1

  preds.append(str(pred))
  labels.append(str(label))
print('test_accuracy=', acc)

import csv
with open(args.output+'/test_results.txt', 'w') as f:
  f.write(str(output[2]))
  f.write(str(args))

header = ['preds', 'targets']
data = [ [str(pred), str(label)] for pred, label in zip(preds, labels) ]

with open(args.output+'/preds_labels.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)