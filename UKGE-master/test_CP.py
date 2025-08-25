import tensorflow as tf
import os
from testtrainer import Trainer
from data import Data
import param
from list import ModelList
import random
import numpy as np
import re
import pandas as pd

def set_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(model_path=None):
    model_path = model_path
    model_path = model_path.replace("\\", "/")
    match = re.search(r"trained_models/(?P<dataset>[^/]+)/(?P<model>[^/_]+)_(?P<date>\d+)_(?P<seed>\d+)/model\.bin-(?P<epoch>\d+)", model_path)
    if match:
        param.whichdata = match.group('dataset')
        data_dir = f"./data/{param.whichdata}"
        param.whichmodel = ModelList(match.group('model'))
        param.seed = int(match.group('seed'))
        set_seed(param.seed)
        epoch = int(match.group('epoch'))

    model_dir = os.path.dirname(model_path)
    model_prefix = model_path

    file_train = os.path.join(data_dir, 'train.tsv')
    file_val = os.path.join(data_dir, 'val.tsv')
    file_test = os.path.join(data_dir, 'test.tsv')
    file_test_with_neg = os.path.join(data_dir, 'test_with_neg.tsv')
    file_psl = os.path.join(data_dir, 'softlogic.tsv')

    param.learning_rate = 0.001
    param.batch_size = 128
    param.val_save_freq = 10
    param.dim = 128
    param.neg_per_pos = 10
    param.reg_scale = 0.0005
    param.n_psl = 0
    param.seed = 0
    param.coverage_pos = 0
    param.sharpness_pos = 0
    param.qhat = 0
    param.coverage_neg = 0
    param.sharpness_neg = 0
    param.neg_qhat = 0
    param.coverage_combined = 0
    param.sharpness_combined = 0
    param.qhat_combined = 0

    this_data = Data()
    this_data.load_data(file_train, file_val, file_test, file_test_with_neg, file_psl)

    m_train = Trainer()
    m_train.build(this_data, model_dir)


    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_prefix)

        valid_mse = m_train.get_val_loss(epoch=epoch, sess=sess)

        test_mse_with_neg = m_train.get_test_with_neg_loss(epoch=epoch, sess=sess)
        
        filename = f"UKGE_{param.whichdata}_{match.group('model')}_con_large.csv"
        print(f"Writing to {filename}")
        new_row = {
            'coverage_pos': param.coverage_pos,
            'sharpness_pos': param.sharpness_pos,
            'qhat': param.qhat,
            'coverage_neg': param.coverage_neg,
            'sharpness_neg': param.sharpness_neg,
            'neg_qhat': param.neg_qhat,
            'coverage_combined': param.coverage_combined,
            'sharpness_combined': param.sharpness_combined,
            'qhat_combined': param.qhat_combined
        }

        if os.path.exists(filename):
            df = pd.read_csv(filename)
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        else:
            df = pd.DataFrame([new_row])

        df.to_csv(filename, index=False)


base_dir = './trained_models'

for dataset in os.listdir(base_dir):
    #if dataset == 'cn15k':
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.isdir(dataset_path):
            for model_folder in os.listdir(dataset_path):
                model_folder_path = os.path.join(dataset_path, model_folder)
                if os.path.isdir(model_folder_path):
                    max_epoch = -1
                    best_model_path = None
                    for file in os.listdir(model_folder_path):
                        if file.startswith('model.bin-') and file.endswith('.index'):
                            try:
                                epoch = int(file[10:].split('.')[0])
                                if epoch > max_epoch:
                                    max_epoch = epoch
                                    best_model_path = os.path.join(model_folder_path, file[:-6])
                            except ValueError:
                                continue
                    if best_model_path:
                        print(f"python test.py --model_path {best_model_path}")
                        main(best_model_path)
                    else:
                        print(f"No model found in {model_folder_path}")
        else:
            print(f"No model found in {dataset_path}")
print("Done")
