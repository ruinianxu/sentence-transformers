'''
This script is for generating csv file based on generated datset from AI2THOR.
It collects each pair of natural language in each task and assign 1 (highest score) to them.
'''
import csv
import os
import random
import collections

ROOT = "/home/ruinian/IVALab/Project/TaskGrounding/AI2THOR_DATA_GENERATION/data/nl"
column_format = ['split', 'genre', 'dataset', 'year', 'sid', 'score', 'sentence1', 'sentence2']

with open("../datasets/gt_rt.csv", 'wt') as out_file:
    tsv_writer = csv.DictWriter(out_file, fieldnames=column_format, delimiter='\t', quoting=csv.QUOTE_NONE)
    # tsv_writer = csv.DictWriter(out_file, fieldnames=column_format)
    tsv_writer.writeheader()

    cnt = 0
    for task_folder in os.listdir(ROOT):
        sentences_list = []
        for instance_folder in os.listdir(os.path.join(ROOT, task_folder)):
            if instance_folder == 'task_description.json':
                continue

            with open(os.path.join(ROOT, task_folder, instance_folder, 'natural_language.txt'), 'r') as f:
                line = f.readline()
                sentences_list.append(line)

        sentences_pair = []
        if len(sentences_list) > 2000:
            random.shuffle(sentences_list)
        sentences_list = sentences_list[:2000]

        for i in range(len(sentences_list)):
            for j in range(i+1, len(sentences_list)):
                sentence1 = sentences_list[i]
                sentence2 = sentences_list[j]

                if sentence1 != sentence2:
                    sentences_pair.append([sentence1, sentence2])

        # shuffle the collected sentences
        random.shuffle(sentences_pair)

        for i in range(len(sentences_pair)):
            if i < int(len(sentences_pair) * 0.8):
                # row['split'] = 'train'
                tsv_writer.writerow({'split': 'train', 'genre': 'robotic_task', 'dataset':'gt_rt', 'year': '2021',
                                      'sid': str(cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0], 'sentence2': sentences_pair[i][1]})
            elif i < int(len(sentences_pair) * 0.9):
                # row['split'] = 'dev'
                tsv_writer.writerow({'split': 'dev', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                     'sid': str(cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0],
                                     'sentence2': sentences_pair[i][1]})
            else:
                # row['split'] = 'test'
                tsv_writer.writerow({'split': 'test', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                     'sid': str(cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0],
                                     'sentence2': sentences_pair[i][1]})
            cnt += 1