"""
Current working directory: Project root dir

=== usage
python run/run.py -m DM --data cn15k --lr 0.01 --batch_size 300
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

'''if '.\\src' not in sys.path:
    sys.path.append('.\\src')

if '.\\' not in sys.path:
    sys.path.append('.\\')'''

import os
from os.path import join
from data import Data

from trainer import Trainer
from list import ModelList
import datetime

import argparse
import param
import tensorflow as tf
import random
import numpy as np
import wandb


def get_model_identifier(whichmodel):
    prefix = whichmodel.value
    now = datetime.datetime.now()
    date = '%02d%02d' % (now.month, now.day)  # two digits month/day
    identifier = prefix + '_' + date + '_' + str(param.seed)
    return identifier

def set_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        set_seed(config.seed)
        param.verbose = "False"
        param.whichdata = config.data
        param.whichmodel = ModelList(config.model)
        param.n_epoch = int(config.epoch)
        param.learning_rate = float('0.001')
        param.batch_size = int('128')
        param.val_save_freq = int(config.save_freq)
        param.dim = int('128')
        param.neg_per_pos = int('10')
        param.reg_scale = float('0.0005')
        param.seed = config.seed

        # path to save
        identifier = get_model_identifier(param.whichmodel)
        save_dir = join('./trained_models', param.whichdata, identifier)  # the directory where we store this model
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        print('Trained models will be stored in: ', save_dir)

        # input files
        data_dir = join('./data', config.data)
        file_train = join(data_dir, 'train.tsv')  # training data
        file_val = join(data_dir, 'val.tsv')  # validation datan
        file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
        file_test = join(data_dir, 'test.tsv') #test data
        file_test_with_neg = join(data_dir, 'test_with_neg.tsv') #test data with negative samples
        print('file_psl: %s' % file_psl)

        more_filt = [file_val, join(data_dir, 'test.tsv')]
        print('Read train.tsv from', data_dir)

        # load data
        this_data = Data()
        this_data.load_data(file_train=file_train, file_val=file_val, file_test=file_test, file_test_with_neg=file_test_with_neg, file_psl=file_psl)
        for f in more_filt:
            this_data.record_more_data(f)
        this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

        m_train = Trainer()
        m_train.build(this_data, save_dir)

        # Model will be trained, validated, and saved in './trained_models'
        ht_embedding, r_embedding = m_train.train(epochs=param.n_epoch, save_every_epoch=param.val_save_freq,
                                                lr=param.learning_rate,
                                                data_dir=param.data_dir(),
                                                wandb=wandb)
        
def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'grid',  # Choose 'random', 'grid', or 'bayes'
        'parameters': {
            'seed': {'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  #, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  # Sweep seeds
            'data': {'values': ['cn15k']},      #, 'cn15k, nl27k, ppi5k'
            'model': {'values': ['logi']},      #, 'logi'
            'epoch': {'values': [3000]},          #, 50, 100
            'save_freq': {'values': [10]},      #, 20
        }
    }
    date_str = datetime.datetime.now().strftime("%m%d")
    project_name = f"{'UKGE_ori'}_{sweep_config['parameters']['data']['values'][0]}_{sweep_config['parameters']['model']['values'][0]}_{date_str}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=sweep_train)

if __name__ =="__main__":
    main()