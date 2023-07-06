import random
import pandas as pd
import json
from tqdm import tqdm
from argparse import ArgumentParser
import re

parser = ArgumentParser()
parser.add_argument("--size", help="size of dataset wanted", required=True)
parser.add_argument("--set", help="set name e.g.train, dev", required=True)
parser.add_argument("--template_file", help="path to templates", default='templates.csv')
parser.add_argument("--output_path", help="path to output questions", default='mixed_base_')


args = parser.parse_args()
size = int(args.size)
sets = args.set
size_small_int = 0.25 * size
size_large_int = 0.25 * size + size_small_int
size_tenths = 0.25 * size + size_large_int
size_hundredths = 0.25 * size + size_tenths

df = pd.read_csv(args.template_file)
n_templates = len(df['templates'])

# generate a question 
def q_a_e_generator(q, a, size_small_int, size_large_int, size_tenths, count):
    answer = -1.2
    k=0
    # fix length of numbers generated
    while answer <= 0 or len(str(answer))>15:
        k +=1
        if count < size_small_int:
            # Generate small integers
            num1, num2, num3 = random.randint(1,999), random.randint(1,999), random.randint(1,999)
        elif count >= size_small_int and count < size_large_int:
            # Generate large integers
            num1, num2, num3 = random.randint(1000,999999), random.randint(1000,999999), random.randint(1000,999999)
        elif count >= size_large_int and count < size_tenths:
            # Generate 1dp
            num1, num2, num3 = round(random.random()*(10**random.randint(0,3)),1), round(random.random()*(10**random.randint(0,3)),1), round(random.random()*(10**random.randint(0,3)),1)
        else :
            # Generate 2dp
            num1, num2, num3 = str(round(random.random()*(10**random.randint(0,3)),2)), str(round(random.random()*(10**random.randint(0,3)),2)), str(round(random.random()*(10**random.randint(0,3)),2))
            length1, length2, length3 = len(re.findall('\.\d+', num1)[0]), len(re.findall('\.\d+', num2)[0]), len(re.findall('\.\d+', num3)[0])
            if length1!=3:
                num1+='0'*(3-length1)
            if length2!=3:
                num2+='0'*(3-length2)
            if length3!=3:
                num3+='0'*(3-length3)
        num1, num2, num3 = str(num1), str(num2), str(num3)
        question = q.replace('num1', num1).replace('num2', num2).replace('num3', num3)
        eqn = str(a.replace('num1', num1).replace('num2', num2).replace('num3', num3))
        try:
            answer = eval(eqn)
        except:
            answer = -1.2
    return question, answer, eqn


# Generate questions that are size modulo # of templates
repeats = int(size // n_templates)
random_size = int(size % n_templates)
sample = random.sample([i for i in range(0, n_templates)], random_size)
# Creates text file of dictionary
with open(f'{args.output_path}_{size}.{sets}', 'w') as f:
    count = 0
    if repeats != 0:
        for i in tqdm(range(repeats)):
            for q, a in zip(df['templates'], df['answers']):
                dict_q_a = {}
                question, answer, eqn = q_a_e_generator(q, a, size_small_int, size_large_int, size_tenths, count)
                count += 1
                dict_q_a['question']=question
                dict_q_a['answer']=str(float(answer))
                dict_q_a['equation']=eqn
                f.write(json.dumps(dict_q_a)+'\n')
    
    for i in sample:
        q = df.templates[i]
        a = df.answers[i]
        dict_q_a = {}
        question, answer, eqn = q_a_e_generator(q, a, size_small_int, size_large_int, size_tenths, count)
        count += 1
        dict_q_a['question']=question
        dict_q_a['answer']=str(float(answer))
        dict_q_a['equation']=eqn
        f.write(json.dumps(dict_q_a)+'\n')
        
print('Data generated!')