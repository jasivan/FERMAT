import os
import json
import csv
import re
import argparse
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--model_checkpoint',
                    required=True,
                    help='path to model checkpoint')
parser.add_argument('--evaluation_set',
                    required=True,
                    help='path to evaluation set as json')
parser.add_argument('--output_name',
                    required=True,
                    help='file name to save results')
args = parser.parse_args()

from transformers import T5Tokenizer, T5ForConditionalGeneration, AutoTokenizer
print('code started')

model = T5ForConditionalGeneration.from_pretrained(args.model_checkpoint).to("cuda")
print('model loaded')
tokenizer = T5Tokenizer.from_pretrained(args.model_checkpoint, device_map="auto")
print('Tokenizer loaded')

model_name = args.model_checkpoint.split('/')[1]
if os.path.exists(f'{args.output_name}') == False:
  os.mkdir(f'{args.output_name}')

results = {model_name:['macro', 'a+b', 'a-b', 'a*b', 'a/b', '(a+b)-c', 'a*(b+c)', '(a+b)/c', 'a*(b-c)', '(a-b)/c']}

for test in os.listdir(args.evaluation_set):
    op_dict = {'macro':0, 'a+b':0, 'a-b':0, 'a*b':0, 'a/b':0, '(a+b)-c':0, 'a*(b+c)':0, '(a+b)/c':0, 'a*(b-c)':0, '(a-b)/c':0}
    op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[1111, 154, 162, 113, 102, 190, 100, 90, 100, 100])]
    questions, targets, preds= [], [], []
    name = str(test).strip('.test')
    print(name)
    with open(f'{args.evaluation_set}/{test}', 'r') as f:
        for line in f:
            d = json.loads(line)
            questions.append(d['question'])
            targets.append(d['answer'])
    input_ids = tokenizer(questions, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")
    print('input tokenised')
    outputs = model.generate(input_ids, max_new_tokens=16).to("cuda")
    outputs = tokenizer.batch_decode(outputs)

    k = 0
    for output, question, targ in tqdm(zip(outputs, questions, targets)):
        prediction = output.replace('\n', '').replace('<pad>', '').replace('</s>', '').replace('<s>', '').replace(' ', '').replace(question, '')
        pred_num = re.findall('(\d+\.\d+|\d+)', prediction)
        if len(pred_num) > 0:
            pred = pred_num[-1]
        else:
            pred = prediction
        preds.append(prediction)

        try:
            if float(pred) == float(targ):
                    op_dict['macro'] +=1
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
        except:
            pass
        k+=1
    op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[1111, 154, 162, 113, 102, 190, 100, 90, 100, 100])]
    results[name] = op_acc
    header = ['preds', 'targets']
    data = [ [str(pred), str(label)] for pred, label in zip(preds, targets) ]
    with open(f'{args.output_name}/{name}-preds_labels.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

print(results)
df = pd.DataFrame(data=results)
df.to_csv(f'{args.output_name}/summary', index=False, mode='w')