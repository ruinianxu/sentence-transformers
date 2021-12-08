'''
This script is for generating csv file based on generated datset from AI2THOR.
It collects each pair of explicit task descriptions in each task and assign 1 (highest score) to them.
Meanwhile this is a simple version which gaurantee that sentences in each sentence pair
are only different in verb and the format.
'''
import csv
import os
import random
import collections

ROOT = "/home/ruinian/IVALab/Project/TaskGrounding/AI2THOR_DATA_GENERATION/data/nl_simple"
column_format = ['split', 'genre', 'dataset', 'year', 'sid', 'score', 'sentence1', 'sentence2']

with open("../datasets/gt_rt_exonly_simple_small_sample.csv", 'wt') as out_file:
    tsv_writer = csv.DictWriter(out_file, fieldnames=column_format, delimiter='\t', quoting=csv.QUOTE_NONE)
    tsv_writer.writeheader()

    pos_cnt = 0

    task_sentences = collections.defaultdict(list)

    # generate the positive samples
    for task_folder in os.listdir(ROOT):
        ex_sentences_list = collections.defaultdict(list)

        for instance_folder in os.listdir(os.path.join(ROOT, task_folder)):
            if instance_folder == 'task_description.json':
                continue

            nl_type = None
            with open(os.path.join(ROOT, task_folder, instance_folder, 'natural_language_type.txt'), 'r') as f:
                nl_type = f.readline()
            if nl_type == 'implicit':
                continue

            objs = []
            with open(os.path.join(ROOT, task_folder, instance_folder, 'natural_language_objects.txt'), 'r') as f:
                line = f.readline().split()
                for obj in line:
                    objs.append(obj)
            objs = tuple(objs)

            with open(os.path.join(ROOT, task_folder, instance_folder, 'natural_language.txt'), 'r') as f:
                line = f.readline()
                ex_sentences_list[objs].append(line)
                task_sentences[task_folder].append(line)


        sentences_pair = []
        for key in ex_sentences_list.keys():
            sentence_list = ex_sentences_list[key]
            for i in range(len(sentence_list)):
                for j in range(len(sentence_list)):
                    sentence1 = sentence_list[i]
                    sentence2 = sentence_list[j]

                    if sentence1 != sentence2:
                        sentences_pair.append([sentence1, sentence2])

        # shuffle the collected sentences
        random.shuffle(sentences_pair)
        sentences_pair = sentences_pair[:10000]

        for i in range(len(sentences_pair)):
            if i < int(len(sentences_pair) * 0.8):
                # row['split'] = 'train'
                tsv_writer.writerow({'split': 'train', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                     'sid': str(pos_cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0],
                                     'sentence2': sentences_pair[i][1]})
            elif i < int(len(sentences_pair) * 0.9):
                # row['split'] = 'dev'
                tsv_writer.writerow({'split': 'dev', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                     'sid': str(pos_cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0],
                                     'sentence2': sentences_pair[i][1]})
            else:
                # row['split'] = 'test'
                tsv_writer.writerow({'split': 'test', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                     'sid': str(pos_cnt).zfill(8), 'score': '5.0', 'sentence1': sentences_pair[i][0],
                                     'sentence2': sentences_pair[i][1]})
            pos_cnt += 1

    # generate the negative samples
    neg_cnt = pos_cnt
    tasks = list(task_sentences.keys())

    while neg_cnt < 2*pos_cnt:
        task1 = random.choice(tasks)
        task2 = None
        while task1 == task2 or task2 is None:
            task2 = random.choice(tasks)

        sentence1 = random.choice(task_sentences[task1])
        sentence2 = random.choice(task_sentences[task2])

        if neg_cnt < int(2 * pos_cnt * 0.8):
            tsv_writer.writerow({'split': 'train', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                 'sid': str(neg_cnt).zfill(8), 'score': '0.0', 'sentence1': sentence1,
                                 'sentence2': sentence2})
        elif neg_cnt < int(2 * pos_cnt * 0.9):
            tsv_writer.writerow({'split': 'dev', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                 'sid': str(neg_cnt).zfill(8), 'score': '0.0', 'sentence1': sentence1,
                                 'sentence2': sentence2})
        else:
            tsv_writer.writerow({'split': 'test', 'genre': 'robotic_task', 'dataset': 'gt_rt', 'year': '2021',
                                 'sid': str(neg_cnt).zfill(8), 'score': '0.0', 'sentence1': sentence1,
                                 'sentence2': sentence2})
        neg_cnt += 1