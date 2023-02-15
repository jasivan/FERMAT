#!/usr/bin/env python
# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import time
import torch
import transformers
from torch.utils.data import SequentialSampler, DataLoader
from torch.utils.data.dataset import TensorDataset
from tqdm import tqdm
import re
import json
import os
import pandas as pd


def processOutput(output_strings):
    output_numbers = []
    for idx, output_string in enumerate(output_strings):
        print('Output:')
        print(output_string)
        indices = [i.start() for i in re.finditer('\?', output_string)]
        number = output_string[indices[-1]+1:]
        number = number.strip(' ')
        print('Extra:')
        print(number)
        try:
            number = float(number)
        except:
            number = float(-1)
        output_numbers.append(str(number))
    return output_numbers


def generateAnswer(inputs, batch_size=20):
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    output_strings = []
    for batch in tqdm(dataloader):
        batch = tuple(t.to(device) for t in batch)
        batch_input_ids, batch_masks = batch
        output = model.generate(input_ids=batch_input_ids, pad_token_id=50256, do_sample=False, max_new_tokens=10, max_input_length=256)
        output.to(torch.device("cpu"))
        for output_example in output:
            string_out = tokenizer.decode(output_example, skip_special_tokens=True)
            output_strings.append(string_out)
    return output_strings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--testset_path', type=str)
    parser.add_argument('-m', '--model_path', type=str)
    parser.add_argument('-g', '--gpt2_path', type=str)
    parser.add_argument('-p', '--pipeline', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = transformers.AutoModelForCausalLM.from_pretrained(args.model_path).to(device)
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.gpt2_path)
    tokenizer.pad_token = tokenizer.eos_token
    right_cnt = 0
    input_prompts = []
    correct_results = []
    tic = time.time()
    cnt = 0

    eval_results = {}
    for category_json in os.listdir(args.testset_path):
        print(category_json)
        with open(f'{args.testset_path}/{category_json}', "r") as infile:
            lines =  infile.readlines()
            for line in lines[:50]:
                line = json.loads(line)
                input_prompt = line['question'].replace("\n", "")
                input_answer = line['answer'].strip("\n").replace(' ', '')
                input_prompts.append(input_prompt)
                correct_results.append(str(input_answer))
                # line = infile.readline()
                cnt += 1

        inputs = tokenizer.batch_encode_plus(
            input_prompts, add_special_tokens=False, return_tensors="pt", padding=True,
            truncation=True)
        outputs = generateAnswer(inputs, batch_size=4)
        results = processOutput(outputs)
        for idx in range(len(results)):
            # if results[idx] != -1.0:
                # print('pred:', results[idx])
                # print('target:', correct_results[idx])
            if results[idx] == correct_results[idx]:
                right_cnt += 1
        accuracy = right_cnt / len(results)
        print(accuracy)
        eval_results[category_json.strip('.json')] = [accuracy, right_cnt]
        # print("Elapsed time: {}".format(time.time() - tic))
        # print("Accuracy score: {}".format(accuracy))
    
    
    print(eval_results)
    df = pd.DataFrame(data=eval_results)
    df.to_csv(f'eval_results_gpt2.csv', index=False)
    print('done')