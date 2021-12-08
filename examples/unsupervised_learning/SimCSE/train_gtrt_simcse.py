from torch.utils.data import DataLoader
import math
from sentence_transformers import models, losses
from sentence_transformers import LoggingHandler, SentenceTransformer, util, InputExample
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
import logging
from datetime import datetime
import os
import gzip
import csv
import numpy as np

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Training parameters
model_name = 'distilbert-base-uncased'
train_batch_size = 16
num_epochs = 4
max_seq_length = 32

# Save path to store our model
model_save_path = 'output/training_stsb_simcse-{}-{}-{}'.format(model_name, train_batch_size, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

#Check if dataset exsist. If not, download and extract  it
dataset_path = '/home/ruinian/IVALab/Project/TaskGrounding/sentence-transformers/datasets/gt_rt_exonly_simple_small_sample.csv'

# Here we define our SentenceTransformer model
word_embedding_model = models.Transformer(model_name, max_seq_length=max_seq_length)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

train_samples = []
dev_samples = []
test_samples = []
# with gzip.open(sts_dataset_path, 'rt', encoding='utf8') as fIn:
average_scores = 0
num_samples = 0
with open(dataset_path, newline='') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t', quoting=csv.QUOTE_NONE)
    for row in reader:
        # score = float(row['score']) / 5.0  # Normalize score to range 0 ... 1

        if row['split'] == 'dev':
            label = np.random.uniform(low=0.99, high=1.0)
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=label)
            dev_samples.append(inp_example)
        elif row['split'] == 'test':
            label = np.random.uniform(low=0.99, high=1.0)
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']], label=label)
            test_samples.append(inp_example)
        else:
            inp_example = InputExample(texts=[row['sentence1'], row['sentence2']])
            train_samples.append(inp_example)


dev_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, batch_size=train_batch_size, name='sts-dev')
test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, batch_size=train_batch_size, name='sts-test')

# We train our model using the MultipleNegativesRankingLoss
train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=train_batch_size, drop_last=True)
train_loss = losses.MultipleNegativesRankingLoss(model)

warmup_steps = math.ceil(len(train_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
evaluation_steps = int(len(train_dataloader) * 0.1) #Evaluate every 10% of the data
logging.info("Training sentences: {}".format(len(train_samples)))
logging.info("Warmup-steps: {}".format(warmup_steps))
logging.info("Performance before training")
dev_evaluator(model)

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=dev_evaluator,
          epochs=num_epochs,
          evaluation_steps=evaluation_steps,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          optimizer_params={'lr': 5e-5},
          use_amp=True         #Set to True, if your GPU supports FP16 cores
          )

##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################


model = SentenceTransformer(model_save_path)
test_evaluator(model, output_path=model_save_path)
