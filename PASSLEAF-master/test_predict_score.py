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
import matplotlib.pyplot as plt
import math
import pickle


def set_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def plot_test_score_distribution(results, model_name="UKGE_logi_m2"):
    for dataset_name, seeds_data in sorted(results.items()):
        num_seeds = len(seeds_data)
        cols = 2  
        rows = math.ceil(num_seeds / cols)
        fig, axes = plt.subplots(rows, cols * 2, figsize=(8 * cols * 2, 5 * rows), sharex=True, sharey=True)
        axes = np.array(axes).reshape(rows, cols * 2)

        for idx, (seed, data) in enumerate(sorted(seeds_data.items())):
            row = idx // cols
            col_left = (idx % cols) * 2
            col_right = col_left + 1

            ax_pos = axes[row][col_left]
            ax_neg = axes[row][col_right]

            pos_scores = np.array(data['test_scores_pos'])
            neg_scores = np.array(data['test_scores_neg'])

            ax_pos.hist(pos_scores, bins=np.linspace(0, 1, 21), color='skyblue', edgecolor='black', alpha=0.7)
            ax_pos.set_xlim(0, 1)
            ax_pos.set_title(f"Seed {seed} — Positive")
            ax_pos.set_xlabel("Score")
            ax_pos.set_ylabel("Count")

            ax_neg.hist(neg_scores, bins=np.linspace(0, 1, 21), color='lightcoral', edgecolor='black', alpha=0.7)
            ax_neg.set_xlim(0, 1)
            ax_neg.set_title(f"Seed {seed} — Negative")
            ax_neg.set_xlabel("Score")
            ax_neg.set_ylabel("Count")

        total_plots = rows * cols
        for j in range(len(seeds_data), total_plots):
            row = j // cols
            col_left = (j % cols) * 2
            col_right = col_left + 1
            fig.delaxes(axes[row][col_left])
            fig.delaxes(axes[row][col_right])

        fig.suptitle(f"Test Score Distributions (Positive vs Negative) — {dataset_name} — {model_name}", fontsize=18)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        plt.savefig(f"TestScoreDist_PosNeg_{dataset_name}_{model_name}.png", dpi=300)
        plt.show()



def main(model_path=None):
    global results

    model_path = model_path.replace("\\", "/")
    match = re.search(r"trained_models/(?P<dataset>[^/]+)/UKGE_logi_m2_(?P<date>\d+)_(?P<seed>\d+)/model\.bin-(?P<epoch>\d+)", model_path)
    if match:
        param.whichdata = match.group('dataset')
        data_dir = f"./data/{param.whichdata}"
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

    param.n_epoch = int(600)
    param.learning_rate = float('0.001')
    param.batch_size = int('512')
    param.val_save_freq = int(40)
    param.dim = int('512')
    param.neg_per_pos = int('10')
    param.reg_scale = float('0.0005')
    param.n_psl = 0
    param.coverage_pos = 0
    param.sharpness_pos = 0
    param.qhat = 0
    param.coverage_neg = 0
    param.sharpness_neg = 0
    param.neg_qhat = 0
    param.coverage_combined = 0
    param.sharpness_combined = 0
    param.qhat_combined = 0
    param.normalizing_coverage_pos = 0
    param.normalizing_sharpness_pos = 0
    param.normalizing_qhat = 0
    param.normalizing_coverage_neg = 0
    param.normalizing_sharpness_neg = 0
    param.normalizing_neg_qhat = 0
    param.normalizing_coverage_combined = 0
    param.normalizing_sharpness_combined = 0
    param.normalizing_qhat_combined = 0


    param.semisupervised_negative_sample = False
    param.semisupervised_negative_sample_v2 = False
    param.semisupervised_v1 = False
    param.semisupervised_v1_1 = False
    param.semisupervised_v2 = True
    param.semisupervised_v2_2 = False
    param.semisupervised_v2_3 = False
    param.sample_balance_v0 = False
    param.sample_balance_v0_1 = False
    param.sample_balance_for_semisuper_v0 = False
    param.resume_model_path = None
    param.no_train = False
    param.whichmodel = ModelList('UKGE_logi_m2')
    param.confidence_level = 0.9
    param.bin_width = 0.05

    this_data = Data()
    this_data.load_data(file_train, file_val, file_test, file_test_with_neg, file_psl)
    m_train = Trainer()
    m_train.build(this_data, model_dir, psl=False)

    def entropy_uncertainty(scores, dataset):
        if dataset == 'cn15k':
            scores = 0.5 * scores + 0.5
            
        # Entropy uncertainty calculation
        scores = np.clip(scores, 1e-6, 1 - 1e-6)
        entropy = -scores * np.log(scores) - (1 - scores) * np.log(1 - scores)
        uncertainty = np.clip(entropy, 1e-6, None)
        
        # Bernoulli variance calculation
        # variance = scores * (1 - scores)
        # uncertainty = np.clip(variance, 1e-6, None)
        return uncertainty
    
    def conformal_prediction(valid_errors, uncertainty, confidence_level):
        # Calculate the nonconformity scores
        normalized_nonconf_scores = np.abs(valid_errors) / uncertainty
        norm_conf_level = np.ceil((len(normalized_nonconf_scores) + 1) * confidence_level) / len(normalized_nonconf_scores)
        normalized_qhat = np.quantile(normalized_nonconf_scores, norm_conf_level, interpolation="higher")

        return normalized_nonconf_scores, normalized_qhat
    
    def normalizing_calculate_metrics(scores, qhat, dataset, w_batch):
        uncertainty_test = entropy_uncertainty(scores, dataset)

        pred_intervals = [
            [max(0, s - qhat * u), min(1, s + qhat * u)]
                for s, u in zip(scores, uncertainty_test)
            ]
        mean_pred_intervals = np.mean([qhat * u for u in uncertainty_test])
        
        cover = [(1 if lb <= w <= ub else 0) for (lb, ub), w in zip(pred_intervals, w_batch)]
        coverage = sum(cover) / len(cover) if cover else 0
        sharpness = sum(ub - lb for lb, ub in pred_intervals) / len(pred_intervals) if pred_intervals else 0

        return coverage, sharpness, mean_pred_intervals
    
    def collect_bin_info(scores, qhat, dataset, w_batch):
        uncertainty = entropy_uncertainty(scores, dataset)
        pred_intervals = [
            [max(0, s - qhat * u), min(1, s + qhat * u)]
            for s, u in zip(scores, uncertainty)
        ]
        errors = np.abs(scores - w_batch)
        covered = [
            (err, ub - lb)
            for (lb, ub), w, err in zip(pred_intervals, w_batch, errors)
            if lb <= w <= ub
        ]
        if not covered:
            return [], []
        covered_errors = [e for (e, _) in covered]
        interval_lengths = [l for (_, l) in covered]
        return covered_errors, interval_lengths

    
    # def get_dataset_bin_range_and_bins(dataset_name):
    #     config = {
    #         'cn15k': {'max_err': 1, 'n_bins': 30},
    #         'nl27k': {'max_err': 1, 'n_bins': 30},
    #         'ppi5k': {'max_err': 1, 'n_bins': 30}
    #     }
    #     default = {'max_err': 0.5, 'n_bins': 20}
    #     return config.get(dataset_name, default)

    # def generate_bins_by_dataset(dataset_name):
    #     cfg = get_dataset_bin_range_and_bins(dataset_name)
    #     return np.linspace(0, cfg['max_err'], cfg['n_bins'] + 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_prefix)

        valid_mse = m_train.get_val_loss(epoch=epoch, sess=sess)
        
        # Get the Qhat for the positive samples
        valid_errors = np.abs(param.errors)
        pos_scores = param.valid_pos_scores
        uncertainty = entropy_uncertainty(pos_scores, param.whichdata)
        normalized_nonconf_scores, normalized_qhat = conformal_prediction(valid_errors, uncertainty, param.confidence_level)
        
        # Get the Qhat for the negative samples
        errors_neg = param.errors_neg
        uncertainty_neg = entropy_uncertainty(errors_neg, param.whichdata)
        normalized_nonconf_scores_neg, normalized_qhat_neg = conformal_prediction(errors_neg, uncertainty_neg, param.confidence_level)
        
        # Get the Qhat for the combined samples
        normalized_nonconf_scores_combined = np.concatenate((normalized_nonconf_scores, normalized_nonconf_scores_neg))
        norm_conf_level_combined = np.ceil((len(normalized_nonconf_scores_combined) + 1) * param.confidence_level) / len(normalized_nonconf_scores_combined)
        normalized_qhat_combined = np.quantile(normalized_nonconf_scores_combined, norm_conf_level_combined, interpolation="higher")
        
        test_mse_with_neg = m_train.get_test_with_neg_loss(epoch=epoch, sess=sess)
        
        # Get the test triples for the positive samples
        scores_pos = param.test_pos_scores
        pos_coverage, pos_sharpness, pos_mean_pred_intervals = normalizing_calculate_metrics(scores_pos, normalized_qhat, param.whichdata, param.test_pos_label)
        
        # Get the test triples for the negative samples
        scores_neg = param.test_neg_scores
        neg_coverage, neg_sharpness, neg_mean_pred_intervals = normalizing_calculate_metrics(scores_neg, normalized_qhat_neg, param.whichdata, param.test_neg_label)
        
        # Get the test triples for the combined samples
        test_scores_combined = param.test_combined_scores
        combined_coverage, combined_sharpness, combined_pred_intervals = normalizing_calculate_metrics(test_scores_combined, normalized_qhat_combined, param.whichdata, param.test_combined_label)
        
        # fixed_bins = np.arange(0.0, 0.51, param.bin_width) 
        
        # dynamic_bins = generate_bins_by_dataset(param.whichdata)
        
        # bin_centers_pos, avg_pos, count_pos, _, covered_error_pos = collect_bin_info(scores_pos, normalized_qhat, param.whichdata, param.test_pos_label, bin_width=param.bin_width, fixed_bins=fixed_bins)
        # bin_centers_neg, avg_neg, count_neg, _, covered_error_neg = collect_bin_info(scores_neg, normalized_qhat_neg, param.whichdata, param.test_neg_label, bin_width=param.bin_width, fixed_bins=fixed_bins)
        # bin_centers_comb, avg_comb, count_comb, _, covered_error_comb = collect_bin_info(test_scores_combined, normalized_qhat_combined, param.whichdata, param.test_combined_label, bin_width=param.bin_width, fixed_bins=dynamic_bins)
        
        # test_errors_all = np.abs(param.test_combined_scores - param.test_combined_label)
        covered_error_pos, interval_length_pos = collect_bin_info(
            scores_pos,
            normalized_qhat,
            param.whichdata,
            param.test_pos_label
        )   
        covered_error_neg, interval_length_neg = collect_bin_info(
            scores_neg,
            normalized_qhat_neg,
            param.whichdata,
            param.test_neg_label
        )

        covered_error_comb, interval_length_comb = collect_bin_info(
            test_scores_combined,
            normalized_qhat_combined,
            param.whichdata,
            param.test_combined_label
        )
        
        if param.whichdata not in results:
            results[param.whichdata] = {}

        results[param.whichdata][param.seed] = {
            'test_scores_pos': param.test_pos_scores.tolist(),   
            'test_scores_neg': param.test_neg_scores.tolist(),   
        }
            
base_dir = './trained_models'
results = {}
for dataset in os.listdir(base_dir):
    # if dataset == 'cn15k':

    model_name = 'UKGE_logi_m2'

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
                    break # for debug one seed
                else:
                    print(f"No model found in {model_folder_path}")
        # break # for debug one dataset
    else:
        print(f"No model found in {dataset_path}")

result_dir = "./"  

# for file_name in os.listdir(result_dir):
#     if file_name.startswith("results_") and file_name.endswith(".pkl"):
#         parts = file_name.replace(".pkl", "").split("_")
#         dataset = parts[1]

#         with open(os.path.join(result_dir, file_name), "rb") as f:
#             data = pickle.load(f)

#         if dataset not in results:
#             results[dataset] = {}
#         results[dataset] = data[dataset]  


if results:
    dataset_name = param.whichdata
    plot_test_score_distribution(results)
print("Done")