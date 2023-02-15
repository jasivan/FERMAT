import argparse
import os
import re
parser = argparse.ArgumentParser()
parser.add_argument('--route',
                    help='path to train/dev/test set in json format')
args = parser.parse_args()

route = args.route
for files in os.listdir(args.route):
    with open(f'{route}/{files}', 'r') as f:
        lines = f.readlines()
        new_lines = []
        for line in lines:
            new_line = re.sub('\}$', '"}', line)
            new_line = new_line.replace('"answer": ', '"answer": "')
            new_line = new_line.replace('""', '"')
            new_lines.append(new_line)

    with open(f'{route}/{files}', 'w') as f:
        f.writelines(new_lines)