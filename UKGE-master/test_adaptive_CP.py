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

def plot_all_seeds_normalized_only(results):
    def bin_statistics(errors, interval_lengths, bin_num=30):
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

    for dataset_name, model_dict in sorted(results.items()):
        for model_type, seeds_data in sorted(model_dict.items()):
            for mode in ['comb', 'pos', 'neg']:
                num_seeds = len(seeds_data)
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

                    bin_centers, bin_means, bin_counts = bin_statistics(covered_errors, interval_lengths, bin_num=30)
                    x = np.array(bin_centers)
                    y = np.array(bin_means)

                    min_error = covered_errors.min()
                    max_error = covered_errors.max()

                    ax_main.plot(x, y, label='Normalized Conformal Prediction', color='red', marker='s')
                    ax_main.set_title(f"Seed: {seed}")
                    ax_main.set_ylabel("Mean Prediction Interval")
                    ax_main.set_xlim(min_error, max_error)

                    ax_hist.hist(covered_errors, bins=np.linspace(min_error, max_error, 21), color='gray', alpha=0.7)
                    ax_hist.set_xlim(min_error, max_error)
                    ax_hist.set_xlabel("Test Error")
                    ax_hist.set_ylabel("Count")

                    ax_cdf = ax_hist.twinx()
                    sorted_errors = np.sort(covered_errors)
                    if len(sorted_errors) > 0:
                        cdf_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 0.9
                        ax_cdf.plot(sorted_errors, cdf_vals, color='blue', linestyle='--')

                    ax_cdf.set_ylim(0, 1)
                    ax_cdf.set_ylabel("CDF", color='blue')
                    ax_cdf.tick_params(axis='y', labelcolor='blue')
                    ax_cdf.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
                    ax_cdf.text(max_error * 1.01, 0.9, 'CDF = 0.9', color='red', va='center')

                    ax_main.legend(fontsize=8)

                total_plots = rows * cols
                for j in range(len(seeds_data), total_plots):
                    row = (j // cols) * 2
                    col = j % cols
                    fig.delaxes(axes[row][col])
                    fig.delaxes(axes[row + 1][col])

                fig.suptitle(f"{dataset_name} — {mode.upper()} (Normalized CP Only)", fontsize=18)
                fig.tight_layout(rect=[0, 0, 1, 0.97])
                plt.savefig(f"NORMALIZED_CP_{dataset_name}_{model_type}_{mode}_only.png", dpi=300)
                plt.show()


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

def main(model_path=None):
    global results
    
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
    # param.seed = 0
    param.coverage_pos = 0
    param.sharpness_pos = 0
    param.qhat = 0
    param.coverage_neg = 0
    param.sharpness_neg = 0
    param.neg_qhat = 0
    param.coverage_combined = 0
    param.sharpness_combined = 0
    param.qhat_combined = 0
    param.confidence_level = 0.9
    param.bin_width = 0.05

    this_data = Data()
    this_data.load_data(file_train, file_val, file_test, file_test_with_neg, file_psl)

    m_train = Trainer()
    m_train.build(this_data, model_dir)

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

    def generate_bins_by_count(errors, n_bins=20):
    
        """
        Generate bin edges that split the error range [0, max_error] into n_bins equal-width bins.
        """
        max_err = max(errors)
        # print("max_err", max_err)åå
        return np.linspace(0, max_err, n_bins + 1)

    # def collect_bin_info(scores, qhat, dataset, w_batch, bin_width=0.1, fixed_bins=None):
    #     """
    #     For covered test points (i.e., where true label falls inside prediction interval),
    #     calculate average prediction interval length per error bin.
    #     """
    #     # Step 1: Calculate uncertainty
    #     uncertainty = entropy_uncertainty(scores, dataset)

    #     # Step 2: Construct prediction intervals
    #     pred_intervals = [
    #         [max(0, s - qhat * u), min(1, s + qhat * u)]
    #         for s, u in zip(scores, uncertainty)
    #     ]

    #     # Step 3: Compute errors and filter only "covered" samples
    #     errors = np.abs(scores - w_batch)
    #     covered = [
    #         (err, ub - lb)
    #         for (lb, ub), w, err in zip(pred_intervals, w_batch, errors)
    #         if lb <= w <= ub
    #     ]
    #     covered_errors = [e for (e, _) in covered] 
    #     if not covered:
    #         return [], [], [], []

    #     # Step 4: Bin setup
    #     if fixed_bins is not None:
    #         bins = fixed_bins
    #         bin_widths = np.diff(bins)
    #         # Note: Assumes uniform bin_width
    #         bin_width = bin_widths[0] if len(set(bin_widths)) == 1 else None
    #     else:
    #         max_error = max(e for e, _ in covered)
    #         bins = np.arange(0, max_error + bin_width, bin_width)

    #     # Step 5: Assign covered errors to bins
    #     bin_indices = np.digitize([e for e, _ in covered], bins)

    #     # Step 6: Aggregate statistics per bin
    #     avg_lengths = []
    #     bin_centers = []
    #     bin_counts = []

    #     for i in range(1, len(bins)):
    #         lengths = [l for idx, (e, l) in enumerate(covered) if bin_indices[idx] == i]
    #         if lengths:
    #             avg_len = np.mean(lengths)
    #             count = len(lengths)
    #         else:
    #             avg_len = 0
    #             count = 0
    #         avg_lengths.append(avg_len)
    #         center = (bins[i - 1] + bins[i]) / 2  # actual bin center
    #         bin_centers.append(center)
    #         bin_counts.append(count)

    #     return bin_centers, avg_lengths, bin_counts, bins, covered_errors
    
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
            return [], [], []

        covered_errors = [e for (e, _) in covered]
        interval_lengths = [l for (_, l) in covered]
        return covered_errors, interval_lengths, errors.tolist()
    
    def get_dataset_bin_range_and_bins(dataset_name):
        config = {
            'cn15k': {'max_err': 1, 'n_bins': 30},
            'nl27k': {'max_err': 1, 'n_bins': 30},
            'ppi5k': {'max_err': 1, 'n_bins': 30}
        }
        default = {'max_err': 0.5, 'n_bins': 20}
        return config.get(dataset_name, default)

    def generate_bins_by_dataset(dataset_name):
        cfg = get_dataset_bin_range_and_bins(dataset_name)
        return np.linspace(0, cfg['max_err'], cfg['n_bins'] + 1)

    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, model_prefix)
        valid_mse = m_train.get_val_loss(epoch=epoch, sess=sess)

        # Get the Qhat for the positive samples
        valid_errors = np.abs(param.valid_errors)
        pos_scores = param.valid_scores
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
        
        # for plotting  
        # fixed_bins = np.arange(0.0, 0.51, param.bin_width) 
        
        # dynamic_bins = generate_bins_by_dataset(param.whichdata)
        
        # bin_centers_pos, avg_pos, count_pos, _, covered_error_pos = collect_bin_info(scores_pos, normalized_qhat, param.whichdata, param.test_pos_label, bin_width=param.bin_width, fixed_bins=fixed_bins)
        # bin_centers_neg, avg_neg, count_neg, _, covered_error_neg= collect_bin_info(scores_neg, normalized_qhat_neg, param.whichdata, param.test_neg_label, bin_width=param.bin_width, fixed_bins=fixed_bins)
        # bin_centers_comb, avg_comb, count_comb, _, covered_error_comb = collect_bin_info(test_scores_combined, normalized_qhat_combined, param.whichdata, param.test_combined_label, bin_width=param.bin_width, fixed_bins=fixed_bins)
        
        # test_errors_all = np.abs(param.test_combined_scores - param.test_combined_label)
        covered_error_pos, interval_length_pos, _ = collect_bin_info(
            scores_pos,
            normalized_qhat,
            param.whichdata,
            param.test_pos_label,
        )   
        covered_error_neg, interval_length_neg, _ = collect_bin_info(
            scores_neg,
            normalized_qhat_neg,
            param.whichdata,
            param.test_neg_label,
        )

        covered_error_comb, interval_length_comb, _ = collect_bin_info(
                                                                test_scores_combined,
                                                                normalized_qhat_combined,
                                                                param.whichdata,
                                                                param.test_combined_label,
                                                            )
        
        if param.whichdata not in results:
            results[param.whichdata] = {}

        if param.whichmodel.value not in results[param.whichdata]:
            results[param.whichdata][param.whichmodel.value] = {}

        results[param.whichdata][param.whichmodel.value][param.seed] = {
            'covered_error_pos': covered_error_pos,
            'interval_length_pos': interval_length_pos,
            'covered_error_neg': covered_error_neg,
            'interval_length_neg': interval_length_neg,
            'covered_error_comb': covered_error_comb,
            'interval_length_comb': interval_length_comb
        }
        
        with open(f"results_{param.whichdata}_{param.whichmodel.value}.pkl", "wb") as f:
            pickle.dump(results, f)
    

base_dir = './trained_models'
results = {}

for dataset in os.listdir(base_dir):
    # if dataset == 'cn15k':
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
                    # break
                else:
                    print(f"No model found in {model_folder_path}")
        # break   
    else:
        print(f"No model found in {dataset_path}")


# result_dir = "./"  
# results = {}
# cp_results = {}

# for file_name in os.listdir(result_dir):
#     if file_name.endswith(".pkl"):
#         file_path = os.path.join(result_dir, file_name)
#         if file_name.startswith("results_"):
#             parts = file_name.replace(".pkl", "").split("_")
#             dataset = parts[1]
#             model = parts[2]

#             with open(file_path, "rb") as f:
#                 data = pickle.load(f)

#             if dataset not in results:
#                 results[dataset] = {}
#             results[dataset][model] = data[dataset][model]

#         elif file_name.startswith("CP_results_"):
#             parts = file_name.replace(".pkl", "").split("_")
#             dataset = parts[2] 
#             model = parts[3]

#             with open(file_path, "rb") as f:
#                 data = pickle.load(f)

#             if dataset not in cp_results:
#                 cp_results[dataset] = {}
#             cp_results[dataset][model] = data[dataset][model]

# print(f"✅ Loaded {len(results)} datasets from results_")
# print(f"✅ Loaded {len(cp_results)} datasets from CP_results_")


# if results:
#     # plot_all_seeds_combined_only(results, cp_results)
#     plot_all_seeds_normalized_only(results)

# print("Done")
