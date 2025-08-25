import torch
import argparse
from dataset import *
from traintest import test_func, evaluate_ndcg
from utils import load_hr_map
from param import *
import os
import re
import pandas as pd
import numpy as np
import random
from os.path import join
import matplotlib.pyplot as plt
import math
import pickle

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser(description='Test with Uncertainty-aware Conformal Prediction')
    return parser.parse_args(args)

def entropy_uncertainty(scores, dataset):
    """
    Calculate entropy-based uncertainty.
    
    If the dataset is 'cn15k', the scores are linearly transformed,
    then the entropy is computed. Finally, values are clamped to avoid numerical issues.
    """
    if dataset == 'cn15k':
        scores = 0.5 * scores + 0.5
    scores = torch.clamp(scores, 1e-6, 1 - 1e-6)
    entropy = -scores * torch.log(scores) - (1 - scores) * torch.log(1 - scores)
    uncertainty = torch.clamp(entropy, 1e-6, None)
    
    # # Bernoulli variance
    # bernoulli_variance = scores * (1 - scores)
    # uncertainty = torch.clamp(bernoulli_variance, 1e-6, None)
    return uncertainty

def torch_conformal_prediction(valid_errors, uncertainty, confidence_level):
    """
    Calculate nonconformity scores and the corresponding quantile (q_hat) on the GPU.
    
    The nonconformity score is computed as the absolute error divided by uncertainty.
    torch.quantile is used on the GPU directly to obtain the desired quantile,
    thereby avoiding unnecessary GPU-to-CPU transfers.
    """
    # Compute nonconformity scores on GPU.
    nonconf_scores = torch.abs(valid_errors) / uncertainty
    # Get the number of elements.
    num = nonconf_scores.numel()
    # Compute the quantile position.
    # Convert (num + 1) and confidence_level into tensors to avoid type errors.
    norm_conf_level = torch.ceil(
        torch.tensor(num + 1, dtype=torch.float32, device=nonconf_scores.device) * 
        torch.tensor(confidence_level, dtype=torch.float32, device=nonconf_scores.device)
    ) / num
    # Compute the quantile (q_hat) using torch.quantile directly on the GPU.
    normalized_qhat = torch.quantile(nonconf_scores, norm_conf_level.item(), interpolation="higher")
    return nonconf_scores, normalized_qhat

def normalizing_calculate_metrics(scores, qhat, dataset, w_batch):
    """
    Calculate prediction intervals, coverage, and sharpness metrics based on qhat.
    
    Prediction intervals are computed from the provided scores and uncertainty.
    Coverage is the fraction of true labels within the prediction intervals, and
    sharpness is the average width of these intervals.
    """
    uncertainty_test = entropy_uncertainty(scores, dataset)
    lb = torch.clamp(scores - qhat * uncertainty_test, min=0)
    ub = torch.clamp(scores + qhat * uncertainty_test, max=1)
    coverage = ((w_batch >= lb) & (w_batch <= ub)).float().mean().item()
    sharpness = (ub - lb).mean().item()
    new_q_hat = (qhat * uncertainty_test).mean().item()
    return coverage, sharpness, new_q_hat

def collect_bin_info(scores, qhat, dataset, w_batch):
    uncertainty = entropy_uncertainty(scores, dataset)
    pred_intervals = torch.stack([
        torch.clamp(scores - qhat * uncertainty, min=0),
        torch.clamp(scores + qhat * uncertainty, max=1)
    ], dim=1)  # shape: (N, 2)

    errors = torch.abs(scores - w_batch)

    lb = pred_intervals[:, 0]
    ub = pred_intervals[:, 1]
    covered_mask = (w_batch >= lb) & (w_batch <= ub)

    if not covered_mask.any():
        return [], []

    covered_errors = errors[covered_mask]
    interval_lengths = (ub - lb)[covered_mask]

    return covered_errors.cpu().numpy(), interval_lengths.cpu().numpy()


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

def plot_all_seeds_combined_only(results, cp_results):
    fisher_values_all = {
        "cn15k": {'comb': 0.63, 'pos': 0.70, 'neg': 0.29},
        "ppi5k": {'comb': 0.61, 'pos': 0.69, 'neg': 0.09},
        "nl27k": {'comb': 1.00, 'pos': 0.67, 'neg': 0.12},
    }
    qr_values_all = {
        "cn15k": {'comb': 0.66, 'pos': 0.70, 'neg': 0.62},
        "ppi5k": {'comb': 0.29, 'pos': 0.49, 'neg': 0.09},
        "nl27k": {'comb': 0.46, 'pos': 0.86, 'neg': 0.06},
    }

    for dataset_name, seeds_data in sorted(results.items()):
        cp_seeds_data = cp_results.get(dataset_name, {})  # 找对应CP
        fisher_values = fisher_values_all.get(dataset_name.lower())
        qr_values = qr_values_all.get(dataset_name.lower())

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

                bins = np.linspace(covered_errors.min(), covered_errors.max(), 31)  # 30个bin
                bin_centers = (bins[:-1] + bins[1:]) / 2

                bin_means = []
                bin_counts = []
                bin_indices = np.digitize(covered_errors, bins) - 1
                bin_indices[bin_indices == len(bin_centers)] = len(bin_centers) - 1

                for i in range(len(bin_centers)):
                    mask = bin_indices == i
                    if np.any(mask):
                        avg_len = interval_lengths[mask].mean()
                        count = mask.sum()
                    else:
                        avg_len = 0
                        count = 0
                    bin_means.append(avg_len)
                    bin_counts.append(count)

                x = np.array(bin_centers)
                y = np.array(bin_means)

                ax_main.plot(x, y, label='Normalized Conformal Prediction', color='red', marker='s')

                ax_main.axhline(y=fisher_values[mode], color='green', linestyle='--', label='Fisher Prediction')
                ax_main.axhline(y=qr_values[mode], color='blue', linestyle='-.', label='Quantile Regression')

                if seed in cp_seeds_data:
                    cp_data = cp_seeds_data[seed]
                    cp_errors = np.array(cp_data[f'covered_error_{mode}'])
                    cp_intervals = np.array(cp_data[f'interval_length_{mode}'])

                    cp_bin_means = []
                    cp_bin_counts = []
                    cp_bin_indices = np.digitize(cp_errors, bins) - 1
                    cp_bin_indices[cp_bin_indices == len(bin_centers)] = len(bin_centers) - 1

                    for i in range(len(bin_centers)):
                        mask = cp_bin_indices == i
                        if np.any(mask):
                            avg_len = cp_intervals[mask].mean()
                            count = mask.sum()
                        else:
                            avg_len = 0
                            count = 0
                        cp_bin_means.append(avg_len)
                        cp_bin_counts.append(count)

                    ax_main.plot(x, cp_bin_means, label='Conformal Prediction', color='orange', marker='^')

                ax_main.set_title(f"Seed: {seed % 10}")
                ax_main.set_ylabel("Mean Prediction Interval")
                ax_main.set_xlim(x.min(), x.max())  

                
                ax_hist.hist(covered_errors, bins=bins, color='gray', alpha=0.7)
                ax_hist.set_xlim(x.min(), x.max())
                ax_hist.set_xlabel("Test Covered Error")
                ax_hist.set_ylabel("Count")

                ax_cdf = ax_hist.twinx()
                sorted_errors = np.sort(covered_errors)
                if len(sorted_errors) > 0:
                    cdf_vals = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 0.9
                    ax_cdf.plot(sorted_errors, cdf_vals, color='purple', linestyle='--')

                ax_cdf.set_ylim(0, 1)
                ax_cdf.set_ylabel("CDF", color='purple')
                ax_cdf.tick_params(axis='y', labelcolor='purple')
                ax_cdf.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
                ax_cdf.text(x.max() * 1.01, 0.9, 'CDF = 0.9', color='red', va='center')

                ax_main.legend(fontsize=8)

            total_plots = rows * cols
            for j in range(len(seeds_data), total_plots):
                row = (j // cols) * 2
                col = j % cols
                fig.delaxes(axes[row][col])
                fig.delaxes(axes[row + 1][col])

            fig.suptitle(f"{dataset_name.upper()} — {mode.upper()}", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"Box_{dataset_name}_entropy_cp_{mode}_fixedbins_with_fisher_qr_cp.png", dpi=300)
            plt.show()

def plot_all_seeds_normalized_only(results):
    for dataset_name, seeds_data in sorted(results.items()):
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

                bins = np.linspace(covered_errors.min(), covered_errors.max(), 31)
                bin_centers = (bins[:-1] + bins[1:]) / 2

                bin_means = []
                bin_indices = np.digitize(covered_errors, bins) - 1
                bin_indices[bin_indices == len(bin_centers)] = len(bin_centers) - 1

                for i in range(len(bin_centers)):
                    mask = bin_indices == i
                    if np.any(mask):
                        avg_len = interval_lengths[mask].mean()
                    else:
                        avg_len = 0
                    bin_means.append(avg_len)

                x = np.array(bin_centers)
                y = np.array(bin_means)

                ax_main.plot(x, y, label='Normalized Conformal Prediction', color='red', marker='s')
                ax_main.set_title(f"Seed: {seed % 10}")
                ax_main.set_ylabel("Mean Prediction Interval")
                ax_main.set_xlim(x.min(), x.max())

                ax_hist.hist(covered_errors, bins=bins, color='gray', alpha=0.7)
                ax_hist.set_xlim(x.min(), x.max())
                ax_hist.set_xlabel("Test Error")
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

                ax_main.legend(fontsize=8)

            total_plots = rows * cols
            for j in range(len(seeds_data), total_plots):
                row = (j // cols) * 2
                col = j % cols
                fig.delaxes(axes[row][col])
                fig.delaxes(axes[row + 1][col])

            fig.suptitle(f"{dataset_name.upper()} — {mode.upper()} (Normalized CP Only)", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"Box_Normalized_{dataset_name}_cp_only_{mode}.png", dpi=300)
            plt.show()
            


def main(args, model_path, seed):
    global results
    
    # Replace backslashes with forward slashes to ensure consistency.
    model_path = model_path.replace("\\", "/")
    # Extract the dataset name and model identifier from the path.
    match = re.search(r"trained_models/(?P<dataset>[^/]+)/(?P<model>[^/_]+)", model_path)
    if not match:
        print(f"Model path {model_path} format is incorrect!")
        return
    data_name = match.group('dataset')
    task = "mse"

    data_dir = join('./data', data_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load validation and test datasets.
    test_dataset = UncertainTripleDataset(data_dir, 'test.tsv')
    val_dataset = UncertainTripleDataset(data_dir, 'val.tsv')

    # Load the model to the proper device.
    model = torch.load(model_path, map_location=device).to(device)
    confidence_level = 0.9

    if task == 'mse':
        # Run test_func on validation data.
        (_, _, _, _, valid_score, valid_label, valid_neg_score, valid_neg_label) = test_func(val_dataset, device, model, {}, neg_mse_also=True)
        
        valid_score_exp = torch.exp(valid_score)
        valid_errors = valid_score_exp - valid_label
        
        # Compute q_hat for positive samples using entropy uncertainty.
        uncertainty = entropy_uncertainty(valid_score_exp, data_name)
        pos_nonconf, normalized_qhat = torch_conformal_prediction(valid_errors, uncertainty, confidence_level)
        
        ## Calculate nonconformity scores for negative samples
        # randomly select the negative samples
        length = valid_neg_score.shape[0]
        indices = torch.arange(0, length, step=2)
        random_choices = torch.randint(0, 2, (len(indices),))
        selected_indices = indices + random_choices
        valid_neg_score = valid_neg_score[selected_indices]
        
        valid_neg_score_exp = torch.exp(valid_neg_score)
        
        # Compute q_hat for negative samples. Here, error is just valid_neg_score_exp.
        uncertainty_neg = entropy_uncertainty(valid_neg_score_exp, data_name)
        
        neg_nonconf, normalized_qhat_neg = torch_conformal_prediction(valid_neg_score_exp, uncertainty_neg, confidence_level)
        
        # Compute q_hat for combined samples by merging positive and negative nonconformity scores.
        combined_nonconf = torch.cat((pos_nonconf, neg_nonconf))
        num_combined = combined_nonconf.numel()
        norm_conf_level_combined = torch.ceil(
            torch.tensor(num_combined + 1, dtype=torch.float32, device=combined_nonconf.device) *
            torch.tensor(confidence_level, dtype=torch.float32, device=combined_nonconf.device)
        ) / num_combined
        normalized_qhat_combined = torch.quantile(combined_nonconf, norm_conf_level_combined.item(), interpolation="higher")

        # Run test_func on test data.
        (_, _, _, _, test_score, test_label, test_neg_score, test_neg_label) = test_func(test_dataset, device, model, {})
        test_score_exp = torch.exp(test_score)
        test_neg_score_exp = torch.exp(test_neg_score)
        
        # Compute metrics for positive samples.
        pos_coverage, pos_sharpness, pos_mean_qhat = normalizing_calculate_metrics(test_score_exp, normalized_qhat, data_name, test_label)
        # Compute metrics for negative samples.
        neg_coverage, neg_sharpness, neg_mean_qhat = normalizing_calculate_metrics(test_neg_score_exp, normalized_qhat_neg, data_name, test_neg_label)
        # Compute metrics for combined samples.
        combined_scores = torch.cat((test_score_exp, test_neg_score_exp))
        combined_labels = torch.cat((test_label, test_neg_label))
        combined_coverage, combined_sharpness, combined_qhat = normalizing_calculate_metrics(combined_scores, normalized_qhat_combined, data_name, combined_labels)

        covered_error_pos, interval_length_pos = collect_bin_info(
            test_score_exp,
            normalized_qhat,
            data_name,
            test_label
        )   
        covered_error_neg, interval_length_neg = collect_bin_info(
            test_neg_score_exp,
            normalized_qhat_neg,
            data_name,
            test_neg_label
        )

        covered_error_comb, interval_length_comb = collect_bin_info(
            combined_scores,
            normalized_qhat_combined,
            data_name,
            combined_labels
        )
        
        if data_name not in results:
            results[data_name] = {}
            
        results[data_name][seed] = {
            'covered_error_pos': covered_error_pos,
            'interval_length_pos': interval_length_pos,
            'covered_error_neg': covered_error_neg,
            'interval_length_neg': interval_length_neg,
            'covered_error_comb': covered_error_comb,
            'interval_length_comb': interval_length_comb
        }
        
        with open(f"results_{data_name}.pkl", "wb") as f:
            pickle.dump(results, f)

if __name__ == "__main__":
    seed = 0
    set_seed(seed)
    base_dir = './trained_models'
    results = {}
    for dataset in os.listdir(base_dir):
        # if dataset == 'cn15k':
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.isdir(dataset_path):
            for model_folder in os.listdir(dataset_path):
                model_folder_path = os.path.join(dataset_path, model_folder)
                print(f"Testing model at {model_folder_path}")
                main(parse_args(), model_folder_path, seed)
                seed += 1
                set_seed(seed)
    result_dir = "./"  
    results = {}
    cp_results = {}

    # # 读入 Fisher/QR 的 results_
    # for file_name in os.listdir(result_dir):
    #     if file_name.startswith("results_") and file_name.endswith(".pkl"):
    #         parts = file_name.replace(".pkl", "").split("_")
    #         dataset = parts[1]

    #         with open(os.path.join(result_dir, file_name), "rb") as f:
    #             data = pickle.load(f)

    #         if dataset not in results:
    #             results[dataset] = {}
    #         results[dataset] = data[dataset]

    # # 读入 Conformal Prediction 的 CP_results_
    # for file_name in os.listdir(result_dir):
    #     if file_name.startswith("CP_results_") and file_name.endswith(".pkl"):
    #         parts = file_name.replace(".pkl", "").split("_")
    #         dataset = parts[2]  # 注意是 parts[2]

    #         with open(os.path.join(result_dir, file_name), "rb") as f:
    #             data = pickle.load(f)

    #         if dataset not in cp_results:
    #             cp_results[dataset] = {}
    #         cp_results[dataset] = data[dataset]

    # if results and cp_results:
    #     # plot_all_seeds_combined_only(results, cp_results)
    #     plot_all_seeds_normalized_only(results)
    # print("Done")
