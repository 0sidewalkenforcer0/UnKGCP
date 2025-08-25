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
    
def bin_statistics(errors, interval_lengths, bin_num=20):
    errors = np.array(errors)
    interval_lengths = np.array(interval_lengths)
    bins = np.linspace(errors.min(), errors.max(), bin_num + 1)

    bin_centers = [(bins[i] + bins[i + 1]) / 2 for i in range(bin_num)]
    bin_means = []
    bin_counts = []

    bin_indices = np.digitize(errors, bins) - 1
    bin_indices[bin_indices == bin_num] = bin_num - 1

    for i in range(bin_num):
        mask = bin_indices == i
        if np.any(mask):
            avg_len = interval_lengths[mask].mean()
            count = np.sum(mask)
        else:
            avg_len = 0
            count = 0
        bin_means.append(avg_len)
        bin_counts.append(count)

    return bin_centers, bin_means, bin_counts

def plot_all_seeds(results, model_name="UKGE_logi_m2", dataset_name="default_dataset"):
        num_seeds = len(results)
        cols = 5
        rows = math.ceil(num_seeds / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows), sharex=True, sharey=True)
        axes = axes.flatten()

        for idx, (seed, data) in enumerate(sorted(results.items())):
            ax = axes[idx]
            x = data['bin_centers']
            ax.plot(x, data['pos'], label='Positive', color='blue', marker='o')
            ax.plot(x, data['neg'], label='Negative', color='green', marker='x')
            ax.plot(x, data['comb'], label='Combined', color='red', marker='s')
            for xi, cp, cn, cc in zip(x, data['count_pos'], data['count_neg'], data['count_comb']):
                label = f"P={cp}\nN={cn}\nC={cc}"
                ax.text(xi, 0.02, label, fontsize=7, rotation=0, ha='center', va='bottom')

            ax.set_title(f"Seed: {seed}")
            ax.set_xlabel("Test Error")
            ax.set_ylabel("Mean Prediction Interval")

        for j in range(len(results), len(axes)):
            fig.delaxes(axes[j])

        fig.suptitle(f"Model: {model_name}", fontsize=16)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right')
        plt.savefig(f"PASSLEAF_{dataset_name}_entropy_cp_{model_name}_covered.png", dpi=300)
        plt.show()

def plot_all_seeds_combined_only(results):
    for dataset_name, seeds_data in sorted(results.items()):
        for mode in ['comb', 'pos', 'neg']: 
            cols = 2
            rows = math.ceil(num_seeds / cols)
            fig, axes = plt.subplots(rows * 2, cols, figsize=(10 * cols, 6 * rows), sharex=False)
            axes = np.array(axes).reshape(rows * 2, cols)

            for idx, (seed, data) in enumerate(sorted(seeds_data.items())):
                row = (idx // cols) * 2
                col = idx % cols
                ax_main = axes[row][col]
                ax_hist = axes[row + 1][col]

                covered_errors = np.array(data[f'covered_error_{mode}'])
                interval_lengths = np.array(data[f'interval_length_{mode}'])

                bin_centers, bin_means, bin_counts = bin_statistics(
                    covered_errors, interval_lengths, bin_num=30
                )

                x = np.array(bin_centers)
                y = np.array(bin_means)

                ax_main.plot(x, y, label=f'{mode.upper()} (binned)', color='red', marker='s')
                ax_main.set_title(f"Seed: {seed}")
                ax_main.set_ylabel("Mean Prediction Interval")
                ax_main.set_xlim(x.min(), x.max())

                ax_hist.hist(covered_errors, bins=np.linspace(x.min(), x.max(), 21), color='gray', alpha=0.7)
                ax_hist.set_xlim(x.min(), x.max())
                ax_hist.set_xlabel("Test Covered Error")
                ax_hist.set_ylabel("Count")

                ax_cdf = ax_hist.twinx()
                sorted_errors = np.sort(covered_errors)
                if len(sorted_errors) > 0:
                    cdf_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 0.9
                    ax_cdf.plot(sorted_errors, cdf_vals, color='blue', linestyle='--', label='CDF')

                ax_cdf.set_ylim(0, 1)
                ax_cdf.set_ylabel("CDF", color='blue')
                ax_cdf.tick_params(axis='y', labelcolor='blue')
                ax_cdf.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
                ax_cdf.text(x.max() * 1.01, 0.9, 'CDF = 0.9', color='red', va='center')

            total_plots = rows * cols
            for j in range(len(seeds_data), total_plots):
                row = (j // cols) * 2
                col = j % cols
                fig.delaxes(axes[row][col])
                fig.delaxes(axes[row + 1][col])

            fig.suptitle(f"{dataset_name} — {mode.upper()}", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"PASSLEAF_{dataset_name}_cp_{mode}_binned_curve.png", dpi=300)
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
    
    def conformal_prediction(valid_errors, confidence_level):
        # Calculate the nonconformity scores
        nonconf_scores = np.abs(valid_errors)
        conf_level = np.ceil((len(nonconf_scores) + 1) * confidence_level) / len(nonconf_scores)
        qhat = np.quantile(nonconf_scores, conf_level, interpolation="higher")

        return nonconf_scores, qhat

    def calculate_metrics(scores, qhat, dataset, w_batch):

        pred_intervals = [
            [max(0, s - qhat), min(1, s + qhat)] for s in scores
            ]
        
        cover = [(1 if lb <= w <= ub else 0) for (lb, ub), w in zip(pred_intervals, w_batch)]
        coverage = sum(cover) / len(cover) if cover else 0
        sharpness = sum(ub - lb for lb, ub in pred_intervals) / len(pred_intervals) if pred_intervals else 0

        return coverage, sharpness, qhat
    
    def collect_bin_info(scores, qhat, dataset, w_batch):
        pred_intervals = [
            [max(0, s - qhat), min(1, s + qhat)]
            for s in scores
        ]

        errors = np.abs(scores - w_batch)
        interval_lengths = [ub - lb for (lb, ub) in pred_intervals]

        return errors.tolist(), interval_lengths, errors.tolist()

    
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
        nonconf_scores, qhat = conformal_prediction(valid_errors, param.confidence_level)
        
        # Get the Qhat for the negative samples
        errors_neg = param.errors_neg
        nonconf_scores_neg, qhat_neg = conformal_prediction(errors_neg, param.confidence_level)
        
        # Get the Qhat for the combined samples
        nonconf_scores_combined = np.concatenate((nonconf_scores, nonconf_scores_neg))
        conf_level_combined = np.ceil((len(nonconf_scores_combined) + 1) * param.confidence_level) / len(nonconf_scores_combined)
        qhat_combined = np.quantile(nonconf_scores_combined, conf_level_combined, interpolation="higher")
        
        test_mse_with_neg = m_train.get_test_with_neg_loss(epoch=epoch, sess=sess)
        
        # Get the test triples for the positive samples
        scores_pos = param.test_pos_scores
        pos_coverage, pos_sharpness, pos_mean_pred_intervals = calculate_metrics(scores_pos, qhat, param.whichdata, param.test_pos_label)
        
        # Get the test triples for the negative samples
        scores_neg = param.test_neg_scores
        neg_coverage, neg_sharpness, neg_mean_pred_intervals = calculate_metrics(scores_neg, qhat_neg, param.whichdata, param.test_neg_label)
        
        # Get the test triples for the combined samples
        test_scores_combined = param.test_combined_scores
        combined_coverage, combined_sharpness, combined_pred_intervals = calculate_metrics(test_scores_combined, qhat_combined, param.whichdata, param.test_combined_label)
        
        covered_error_pos, interval_length_pos, _ = collect_bin_info(
            scores_pos,
            qhat,
            param.whichdata,
            param.test_pos_label,
        )   
        covered_error_neg, interval_length_neg, _ = collect_bin_info(
            scores_neg,
            qhat_neg,
            param.whichdata,
            param.test_neg_label,
        )

        covered_error_comb, interval_length_comb, _ = collect_bin_info(
            test_scores_combined,
            qhat_combined,
            param.whichdata,
            param.test_combined_label,
        )
        
        if param.whichdata not in results:
            results[param.whichdata] = {}

        results[param.whichdata][param.seed] = {
            'covered_error_pos': covered_error_pos,
            'interval_length_pos': interval_length_pos,
            'covered_error_neg': covered_error_neg,
            'interval_length_neg': interval_length_neg,
            'covered_error_comb': covered_error_comb,
            'interval_length_comb': interval_length_comb
        }

        with open(f"CP_results_{param.whichdata}.pkl", "wb") as f:
            pickle.dump(results, f)
            
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
                    # break # for debug
                else:
                    print(f"No model found in {model_folder_path}")
        # break # for debug
    else:
        print(f"No model found in {dataset_path}")

# result_dir = "./"  

# for file_name in os.listdir(result_dir):
#     if file_name.startswith("CP_") and file_name.endswith(".pkl"):
#         parts = file_name.replace(".pkl", "").split("_")
#         dataset = parts[2]

#         with open(os.path.join(result_dir, file_name), "rb") as f:
#             data = pickle.load(f)

#         if dataset not in results:
#             results[dataset] = {}
#         results[dataset] = data[dataset]  


# if results:
#     dataset_name = param.whichdata
#     # plot_all_seeds(results, model_name=model_name, dataset_name=dataset_name)
#     plot_all_seeds_combined_only(results)
# print("Done")