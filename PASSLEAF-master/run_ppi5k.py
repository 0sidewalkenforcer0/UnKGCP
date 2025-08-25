"""
Current working directory: Project root dir

=== usage
python run/run.py -m DM --data cn15k --lr 0.01 --batch_size 300
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

'''if './src' not in sys.path:
    sys.path.append('./src')

if './' not in sys.path:
    sys.path.append('./')'''

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
        '''parser = argparse.ArgumentParser()
        # required
        parser.add_argument('--data', type=str, default='cn15k',
                            help="the dir path where you store data (train.tsv, val.tsv, test.tsv). Default: ppi5k")
        # optional
        parser.add_argument("--verbose", help="print detailed info for debugging",
                            action="store_true")
        parser.add_argument('-m', '--model', type=str, default='ComplEx_m5_4', help="choose model ('logi' or 'rect'). default: rect")
        parser.add_argument('-d', '--dim', type=int, default=512, help="set dimension. default: 128")
        parser.add_argument('--epoch', type=int, default=3000, help="set number of epochs. default: 100")
        parser.add_argument('--lr', type=float, default=0.001, help="set learning rate. default: 0.001")
        parser.add_argument('--batch_size', type=int, default=512, help="set batch size. default: 1024")
        parser.add_argument('--n_neg', type=int, default=10, help="Number of negative samples per (h,r,t). default: 10")
        parser.add_argument('--save_freq', type=int, default=2,
                            help="how often (how many epochs) to run validation and save tf models. default: 10")
        parser.add_argument('--models_dir', type=str, default='./trained_models',
                            help="the dir path where you store trained models. A new directory will be created inside it.")

        parser.add_argument('--resume_model_path', type=str, default=None,
                            help="the dir path where you stored trained models.")

        parser.add_argument('--no_psl', action='store_true', default=True)
        parser.add_argument('--semisupervised_neg', action='store_true')
        parser.add_argument('--semisupervised_neg_v2', action='store_true')
        parser.add_argument('--semisupervised_v1', action='store_true')
        parser.add_argument('--semisupervised_v1_1', action='store_true')
        parser.add_argument('--semisupervised_v2', action='store_true', default=True, help='enable the pool-based semi-supervised learning.')
        parser.add_argument('--semisupervised_v2_2', action='store_true')
        parser.add_argument('--semisupervised_v2_3', action='store_true')
        parser.add_argument('--sample_balance_v0', action='store_true')
        parser.add_argument('--sample_balance_v0_1', action='store_true')
        parser.add_argument('--sample_balance_for_semisuper_v0', action='store_true')

        parser.add_argument('--no_trail', action='store_true')

        # regularizer coefficient (lambda)
        parser.add_argument('--reg_scale', type=float, default=0.005,
                            help="The scale for regularizer (lambda). Default 0.0005")

        args = parser.parse_args()'''

        # parameters
        param.verbose = False
        param.whichdata = config.data
        param.whichmodel = ModelList(config.model)
        param.n_epoch = int(config.epoch)
        param.learning_rate = float('0.001')
        param.batch_size = int('512')
        param.val_save_freq = int(config.save_freq)  # The frequency to validate and save model
        param.dim = int('512')  # default 128
        param.neg_per_pos = int('10')  # Number of negative samples per (h,r,t). default 10.
        param.reg_scale = float('0.0005')
        param.n_psl = 0 #if args.no_psl else param.n_psl
        param.semisupervised_negative_sample = False #args.semisupervised_neg
        param.semisupervised_negative_sample_v2 = False #args.semisupervised_neg_v2
        param.semisupervised_v1 = False #args.semisupervised_v1
        param.semisupervised_v1_1 = False #args.semisupervised_v1_1
        param.semisupervised_v2 = True #args.semisupervised_v2
        param.semisupervised_v2_2 = False #args.semisupervised_v2_2
        param.semisupervised_v2_3 = False #args.semisupervised_v2_3
        param.sample_balance_v0 = False #args.sample_balance_v0
        param.sample_balance_v0_1 = False #args.sample_balance_v0_1
        param.sample_balance_for_semisuper_v0 = False #args.sample_balance_for_semisuper_v0
        param.resume_model_path = None #args.resume_model_path
        param.no_train = False #args.no_trail
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
        file_test = join(data_dir, 'test.tsv')  # validation datan
        file_test_with_neg = join(data_dir, 'test_with_neg.tsv') #test data with negative samples
        file_psl = join(data_dir, 'softlogic.tsv')  # probabilistic soft logic
        print('file_psl: %s' % file_psl)

        more_filt = [file_val, join(data_dir, 'test.tsv')]
        print('Read train.tsv from', data_dir)

        if not param.no_train:
            # load data
            this_data = Data()
            this_data.load_data(file_train=file_train, file_val=file_val, file_test=file_test, file_test_with_neg=file_test_with_neg, file_psl= file_psl )
            for f in more_filt:
                this_data.record_more_data(f)
            this_data.save_meta_table(save_dir)  # output: idx_concept.csv, idx_relation.csv

            m_train = Trainer()
            m_train.build(this_data, save_dir, psl=(param.n_psl > 0), semisupervised_negative_sample=param.semisupervised_negative_sample)
            # Model will be trained, validated, and saved in './trained_models'
            ht_embedding, r_embedding = m_train.train(epochs=param.n_epoch, save_every_epoch=param.val_save_freq,
                                                lr=param.learning_rate,
                                                data_dir=param.data_dir(), resume_model_path=param.resume_model_path, wandb=wandb)
        else:
            m_train = Trainer()
            m_train.build(None, save_dir, psl=(param.n_psl > 0))
            m_train.test(file_test = file_test, resume_model_path=param.resume_model_path)

def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'grid',  # Choose 'random', 'grid', or 'bayes'
        'parameters': {
            'seed': {'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  #, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  # Sweep seeds
            'data': {'values': ['ppi5k']},      #, 'cn15k, nl27k, ppi5k'
            'model': {'values': ['UKGE_logi_m2']},      #, 'logi'
            'epoch': {'values': [600]},          #, 50, 100
            'save_freq': {'values': [40]},      #, 20
        }
    }
    date_str = datetime.datetime.now().strftime("%m%d")
    project_name = f"{'Passleaf_ori'}_{sweep_config['parameters']['data']['values'][0]}_{sweep_config['parameters']['model']['values'][0]}_{date_str}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=sweep_train)

if __name__ =="__main__":
    main()
