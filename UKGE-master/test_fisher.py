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
from scipy import stats

def set_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def main(model_path=None):
    model_path = model_path
    model_path = model_path.replace("\\", "/")
    #match = re.search(r"trained_models/(?P<dataset>[^/]+)/(?P<model>[^/_]+)_(?P<date>\d+)_(?P<seed>\d+)_((?P<quantile>\d+(\.\d+)?))/model\.bin-(?P<epoch>\d+)", model_path)
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

        '''for var in tf.trainable_variables():
            value = sess.run(var)
            print(f"{var.name}: {value}")'''

        valid_mse = m_train.get_val_loss(epoch=epoch, sess=sess)

        valid_pos_scores = param.valid_pos_scores
        valid_neg_scores = param.errors_neg

        n_pos = len(valid_pos_scores)
        n_neg = len(valid_neg_scores)

        # Compute mean and standard deviation for positive and negative samples
        mean_valid, std_valid = valid_pos_scores.mean(), valid_pos_scores.std(ddof=1)
        mean_valid_neg, std_valid_neg = valid_neg_scores.mean(), valid_neg_scores.std(ddof=1)
        combined_valid_scores = np.concatenate((valid_pos_scores, valid_neg_scores), axis=0)
        n_combined = len(combined_valid_scores)
        mean_combined = combined_valid_scores.mean()
        std_combined = combined_valid_scores.std(ddof=1)

        # Compute t-values for 90% confidence interval
        confidence = 0.90
        alpha = 1 - confidence
        t_value_pos = stats.t.ppf(1 - alpha / 2, df=n_pos - 2)
        t_value_neg = stats.t.ppf(1 - alpha / 2, df=n_neg - 2)
        t_value_combined = stats.t.ppf(1 - alpha / 2, df=n_combined - 2)

        # Compute the Fisher's prediction interval margins
        margin_pos = t_value_pos * std_valid * np.sqrt(n_pos / (n_pos - 1))
        margin_neg = t_value_neg * std_valid_neg * np.sqrt(n_neg / (n_neg - 1))
        margin_combined = t_value_combined * std_combined * np.sqrt(n_combined / (n_combined - 1))

        test_mse_with_neg = m_train.get_test_with_neg_loss(epoch=epoch, sess=sess)

        test_label = param.test_pos_label
        test_neg_label = param.test_neg_label

        test_score = np.full_like(test_label, fill_value=mean_valid)
        lb_pos = np.clip(test_score - margin_pos, a_min=0, a_max=None)
        ub_pos = np.clip(test_score + margin_pos, a_min=None, a_max=1)
        coverage_pos = np.mean((test_label >= lb_pos) & (test_label <= ub_pos))
        sharpness_pos = np.mean(ub_pos - lb_pos)

        test_neg_score = np.full_like(test_neg_label, fill_value=mean_valid_neg)
        lb_neg = np.clip(test_neg_score - margin_neg, a_min=0, a_max=None)
        ub_neg = np.clip(test_neg_score + margin_neg, a_min=None, a_max=1)
        coverage_neg = np.mean((test_neg_label >= lb_neg) & (test_neg_label <= ub_neg))
        sharpness_neg = np.mean(ub_neg - lb_neg)

        combined_test_label = np.concatenate((test_label, test_neg_label), axis=0)
        combined_test_score = np.full_like(combined_test_label, fill_value=mean_combined)
        lb_combined = np.clip(combined_test_score - margin_combined, a_min=0, a_max=None)
        ub_combined = np.clip(combined_test_score + margin_combined, a_min=None, a_max=1)
        coverage_combined = np.mean((combined_test_label >= lb_combined) & (combined_test_label <= ub_combined))
        sharpness_combined = np.mean(ub_combined - lb_combined)
        
        #filename = f"UKGE_{param.whichdata}_{match.group('model')}_{match.group('quantile')}.csv"
        filename = f"UKGE_Fisher_{param.whichdata}_{match.group('model')}.csv"
        print(f"Writing to {filename}")
        new_row = {
            'coverage_pos': coverage_pos,
            'sharpness_pos': sharpness_pos,
            'qhat_pos': margin_pos.item(),
            'coverage_neg': coverage_neg,
            'sharpness_neg': sharpness_neg,
            'qhat_neg': margin_neg.item(),
            'coverage_combined': coverage_combined,
            'sharpness_combined': sharpness_combined,
            'qhat_combined': margin_combined.item()
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

