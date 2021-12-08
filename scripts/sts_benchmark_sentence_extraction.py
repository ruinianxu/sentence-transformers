"""
This script extracts sentences pairs from sts benchmark into a txt file for checking

Usage:
python sts_benchmark_sentence_extraction.py
"""

import gzip
import csv

sts_dataset_path = '../datasets/stsbenchmark.tsv.gz'

out_file = open('./sts_benchmark_sentence_pair.txt', 'w')
with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
    reader = csv.DictReader(fIn, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        out_file.write(row['sentence1'] + ' ' + row['sentence2'] + '\n')
out_file.close()