
import torch
import argparse
from dataset import *
from traintest import test_func, evaluate_ndcg
from utils import load_hr_map
from param import *
from scipy import stats
import os
import re

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

def main(args, model_path):
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
        valid_score = torch.exp(valid_score)
        valid_neg_score = torch.exp(valid_neg_score)
        
        length = valid_neg_score.shape[0]
        if length % 2 != 0:
            valid_neg_score = valid_neg_score[:-1]
            valid_neg_label = valid_neg_label[:-1]
            print("Odd number of negative samples. Removed the last one.")
        indices = torch.arange(0, length, step=2)
        random_choices = torch.randint(0, 2, (len(indices),))
        selected_indices = indices + random_choices
        valid_neg_score = valid_neg_score[selected_indices]
        valid_neg_label = valid_neg_label[selected_indices]
        
        n_pos = len(valid_score)
        n_neg = len(valid_neg_score)

        # Compute mean and standard deviation for positive and negative samples
        mean_valid, std_valid = valid_score.mean(), valid_score.std(unbiased=True)
        mean_valid_neg, std_valid_neg = valid_neg_score.mean(), valid_neg_score.std(unbiased=True)

        combined_valid_scores = torch.cat((valid_score, valid_neg_score))
        n_combined = len(combined_valid_scores)
        mean_combined = combined_valid_scores.mean()
        std_combined = combined_valid_scores.std(unbiased=True)

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

        # Get test scores for positive and negative samples
        _, _, _, _, test_score, test_label, test_neg_score, test_neg_label = test_func(
            test_dataset, device, model, {}
        )

        test_score = torch.exp(test_score)
        test_neg_score = torch.exp(test_neg_score)

        # Calculate prediction intervals and coverage for positive samples
        test_score = torch.full_like(test_label, mean_valid)
        lb_pos = torch.clamp(test_score - margin_pos, min=0)
        ub_pos = torch.clamp(test_score + margin_pos, max=1)
        coverage_pos = ((test_label >= lb_pos) & (test_label <= ub_pos)).float().mean().item()
        sharpness_pos = (ub_pos - lb_pos).mean().item()

        # Calculate prediction intervals and coverage for negative samples
        test_neg_score = torch.full_like(test_neg_label, mean_valid_neg)
        lb_neg = torch.clamp(test_neg_score - margin_neg, min=0)
        ub_neg = torch.clamp(test_neg_score + margin_neg, max=1)
        coverage_neg = ((test_neg_label >= lb_neg) & (test_neg_label <= ub_neg)).float().mean().item()
        sharpness_neg = (ub_neg - lb_neg).mean().item()

        # Calculate prediction intervals and coverage for combined samples
        combined_test_label = torch.cat((test_label, test_neg_label))
        combined_test_score = torch.full_like(combined_test_label, mean_combined)
        lb_combined = torch.clamp(combined_test_score - margin_combined, min=0)
        ub_combined = torch.clamp(combined_test_score + margin_combined, max=1)
        coverage_combined = ((combined_test_label >= lb_combined) & (combined_test_label <= ub_combined)).float().mean().item()
        sharpness_combined = (ub_combined - lb_combined).mean().item()

        # Save evaluation results into CSV file
        filename = f"FISHER_{data_name}_test.csv"
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

if __name__ =="__main__":
    seed = 0
    set_seed(seed)
    base_dir = './trained_models'

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
    print("Done")
