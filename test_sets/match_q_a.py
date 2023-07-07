import json
import os

# iterates through question files
for file in os.listdir('./questions'):
    if '.q' in file:
        name = str(file.strip('.q'))
        Q = []
        A = []
        # stores questions and answers in list
        with open('./questions/'+file) as f:
            for line in f:
                dic = json.loads(line)
                Q.append((dic["index"], dic["question"]))
        with open('./answers/'+name+'.a') as f:
            for line in f:
                dic = json.loads(line)
                A.append((dic["index"], dic["answer"], dic["equation"]))
        # sort questions and answers
        Q.sort(key=lambda x: x[0])
        A.sort(key=lambda x: x[0])
        # write new file with paired questions and answers
        with open(name+'.test', 'w') as f:
            for q, a in zip(Q,A):
                if q[0] == a[0]:
                    dic = {"question":q[1], "answer":a[1], "equation":a[2]}
                    f.write(json.dumps(dic)+'\n')
                else:
                    print('Error in index matching')