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
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='test-pretrained.py [<args>] [-h | --help]'
    )
    #parser.add_argument('--data', type=str, default='nl27k', help="cn15k or nl27k")
    #parser.add_argument('--task', type=str, default='mse', help="mse or ndcg")
    #parser.add_argument('--model_path', default='./trained_models/nl27k/bigumbelbox-4w54lvai.pt', type=str, help="trained model file.")

    return parser.parse_args(args)

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

def plot_all_seeds_combined_only(results):
    fisher_values = {'comb': 0.63, 'pos': 0.70, 'neg': 0.29}
    qr_values = {'comb': 0.66, 'pos': 0.70, 'neg': 0.62}

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

                bin_centers, bin_means, bin_counts = bin_statistics(
                    covered_errors, interval_lengths, bin_num=30
                )

                x = np.array(bin_centers)
                y = np.array(bin_means)

                ax_main.plot(x, y, label='Normalized Conformal Prediction', color='red', marker='s')

                ax_main.axhline(y=fisher_values[mode], color='green', linestyle='--', label='Fisher Prediction')

                ax_main.axhline(y=qr_values[mode], color='blue', linestyle='-.', label='Quantile Regression')

                ax_main.set_title(f"Seed: {seed % 10}")
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
                    ax_cdf.plot(sorted_errors, cdf_vals, color='purple', linestyle='--', label='CDF')

                ax_cdf.set_ylim(0, 1)
                ax_cdf.set_ylabel("CDF", color='purple')
                ax_cdf.tick_params(axis='y', labelcolor='purple')
                ax_cdf.axhline(y=0.9, color='red', linestyle='--', linewidth=1)
                ax_cdf.text(x.max() * 1.01, 0.9, 'CDF = 0.9', color='red', va='center')

                ax_main.legend()

            total_plots = rows * cols
            for j in range(len(seeds_data), total_plots):
                row = (j // cols) * 2
                col = j % cols
                fig.delaxes(axes[row][col])
                fig.delaxes(axes[row + 1][col])

            fig.suptitle(f"{dataset_name} — {mode.upper()}", fontsize=18)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            plt.savefig(f"Box_{dataset_name}_cp_{mode}_with_fisher_qr.png", dpi=300)
            plt.show()

def main(args, model_path):
    global results
    model_path = model_path
    model_path = model_path.replace("\\", "/")
    match = re.search(r"trained_models/(?P<dataset>[^/]+)/(?P<model>[^/_]+)", model_path)
    data_name = match.group('dataset')
    task = "mse"

    data_dir = join('./data', data_name)
    '''if data_name == 'cn15k':
        num_entity = 15000
    else:
        num_entity = 27221'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = UncertainTripleDataset(data_dir, 'test.tsv')
    val_dataset = UncertainTripleDataset(data_dir, 'val.tsv')

    model = torch.load(model_path)

    if task == 'mse':
        _, _, _, _, valid_score, valid_label, valid_neg_score, valid_neg_label = test_func(
            val_dataset, device, model, {}, neg_mse_also=True
        )
        errors = torch.exp(valid_score) - valid_label
        nonconf_scores = torch.abs(errors)

        length = valid_neg_score.shape[0]
        indices = torch.arange(0, length, step=2)
        random_choices = torch.randint(0, 2, (len(indices),))
        selected_indices = indices + random_choices
        valid_neg_score = valid_neg_score[selected_indices]

        errors_neg = torch.exp(valid_neg_score)
        nonconf_scores_neg = torch.abs(errors_neg)

        nonconf_scores_np = nonconf_scores.cpu().numpy()
        nonconf_scores_neg_np = nonconf_scores_neg.cpu().numpy()

        conf_level = np.ceil((len(nonconf_scores_np) + 1) * 0.9) / len(nonconf_scores_np)
        pos_qhat = np.quantile(nonconf_scores_np, conf_level, method="higher")

        conf_level_neg = np.ceil((len(nonconf_scores_neg_np) + 1) * 0.9) / len(nonconf_scores_neg_np)
        neg_qhat = np.quantile(nonconf_scores_neg_np, conf_level_neg, method="higher")

        combined_nonconf_scores_np = np.concatenate((nonconf_scores_np, nonconf_scores_neg_np))
        conf_level_combined = np.ceil((len(combined_nonconf_scores_np) + 1) * 0.9) / len(combined_nonconf_scores_np)
        combined_qhat = np.quantile(combined_nonconf_scores_np, conf_level_combined, method="higher")

        _, _, _, _, test_score, test_label, test_neg_score, test_neg_label = test_func(
            test_dataset, device, model, {}
        )

        test_score_exp = torch.exp(test_score)
        lb_pos = torch.clamp(test_score_exp - pos_qhat, min=0)
        ub_pos = torch.clamp(test_score_exp + pos_qhat, max=1)
        coverage = ((test_label >= lb_pos) & (test_label <= ub_pos)).float().mean().item()
        sharpness = (ub_pos - lb_pos).mean().item()

        test_neg_score_exp = torch.exp(test_neg_score)
        lb_neg = torch.clamp(test_neg_score_exp - neg_qhat, min=0)
        ub_neg = torch.clamp(test_neg_score_exp + neg_qhat, max=1)
        coverage_neg = ((test_neg_label >= lb_neg) & (test_neg_label <= ub_neg)).float().mean().item()
        sharpness_neg = (ub_neg - lb_neg).mean().item()

        combined_score = torch.cat((test_score, test_neg_score))
        combined_label = torch.cat((test_label, test_neg_label))
        combined_score_exp = torch.exp(combined_score)
        lb_combined = torch.clamp(combined_score_exp - combined_qhat, min=0)
        ub_combined = torch.clamp(combined_score_exp + combined_qhat, max=1)
        coverage_combined = ((combined_label >= lb_combined) & (combined_label <= ub_combined)).float().mean().item()
        sharpness_combined = (ub_combined - lb_combined).mean().item()

        # Generate the results
        # Positive
        lb_pos = torch.clamp(test_score_exp - pos_qhat, min=0)
        ub_pos = torch.clamp(test_score_exp + pos_qhat, max=1)
        covered_error_pos = torch.abs(test_label - test_score_exp).cpu().numpy().tolist()
        interval_length_pos = (ub_pos - lb_pos).cpu().numpy().tolist()

        # Negative
        lb_neg = torch.clamp(test_neg_score_exp - neg_qhat, min=0)
        ub_neg = torch.clamp(test_neg_score_exp + neg_qhat, max=1)
        covered_error_neg = torch.abs(test_neg_label - test_neg_score_exp).cpu().numpy().tolist()
        interval_length_neg = (ub_neg - lb_neg).cpu().numpy().tolist()

        # Combined
        lb_combined = torch.clamp(combined_score_exp - combined_qhat, min=0)
        ub_combined = torch.clamp(combined_score_exp + combined_qhat, max=1)
        covered_error_comb = torch.abs(combined_label - combined_score_exp).cpu().numpy().tolist()
        interval_length_comb = (ub_combined - lb_combined).cpu().numpy().tolist()
        
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
        
        with open(f"CP_results_{data_name}.pkl", "wb") as f:
            pickle.dump(results, f)

if __name__ =="__main__":
    seed = 0
    set_seed(seed)
    base_dir = './trained_models'
    results = {}
    
    for dataset in os.listdir(base_dir):
        #if dataset == "ppi5k":
        dataset_path = os.path.join(base_dir, dataset)
        if os.path.isdir(dataset_path):
            for model_folder in os.listdir(dataset_path):
                model_folder_path = os.path.join(dataset_path, model_folder)
                print(f"python test.py --model_path {model_folder_path}")
                main(parse_args(), model_folder_path)
                seed += 1
                set_seed(seed)
    
    if results:
        plot_all_seeds_combined_only(results)
    print("Done")