import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pred_set',
                    required=True,
                    help='path to prediction set')
args = parser.parse_args()
folder_name = args.pred_set.split('/')[-1]
results = {folder_name:['macro', 'a+b', 'a-b', 'a*b', 'a/b', '(a+b)-c', 'a*(b+c)', '(a+b)/c', 'a*(b-c)', '(a-b)/c']}

for file in os.listdir(args.pred_set):
  name = str(file).replace('-preds_labels.csv', '')
  acc = 0
  op_dict = {'macro':0, 'a+b':0, 'a-b':0, 'a*b':0, 'a/b':0, '(a+b)-c':0, 'a*(b+c)':0, '(a+b)/c':0, 'a*(b-c)':0, '(a-b)/c':0}
  op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[1111, 154, 162, 113, 102, 190, 100, 90, 100, 100])]

  df = pd.read_csv(args.pred_set+'/'+str(file))
  for k in range(len(df['preds'])):
        try:
            if float(df['preds'][k]) == float(df['targets'][k]):
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

  op_acc = [round(i*100/j,4) for i,j in zip(op_dict.values(),[1111, 154, 162, 113, 102, 190, 100, 90, 100, 100])]
  results[name] = op_acc
print(results)
df = pd.DataFrame(data=results)
df.to_csv(f'{args.pred_set}/summary', index=False, mode='w')