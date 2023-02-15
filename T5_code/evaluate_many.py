import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-model_dir',
                    required=True,
                    help='dir to model checkpoint')
parser.add_argument('-evaluation_set',
                    required=True,
                    help='path to evaluation set as json')
parser.add_argument('-output_name',
                    required=True,
                    help='file name to save results')
args = parser.parse_args()

model_dir = args.model_dir

for directory in os.listdir(model_dir):
    model_name = directory
    print(model_name)
    with open(args.output_name, mode='a') as f:
        f.write(model_name+'\n')
    list_ckps = os.listdir(model_dir+'/'+directory)
    num_ckps = [int(name.strip('checkpoint-')) for name in list_ckps]
    num_ckps.sort()
    for model_checkpoint in list_ckps:
        if 'checkpoint' in model_checkpoint and str(num_ckps[0]) in model_checkpoint:
            os.system(f'python T5_code/evaluate_from_ckpt.py -model_checkpoint={model_dir}/{directory}/{model_checkpoint} -evaluation_set={args.evaluation_set} -output_name={args.output_name}')

print('done')