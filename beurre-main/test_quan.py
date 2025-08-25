
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
    match = re.search(r"trained_models/(?P<dataset>[^/]+)/(?P<model>[^/_]+).*-(?P<seed>\d+)-(?P<quantile>[\d.]+)\.pt", model_path)
    data_name = match.group('dataset')
    seed = int(match.group("seed"))
    set_seed(seed)
    quantile = float(match.group("quantile"))
    task = "mse"

    data_dir = join('./data', data_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_dataset = UncertainTripleDataset(data_dir, 'test.tsv')
    val_dataset = UncertainTripleDataset(data_dir, 'val.tsv')

    model = torch.load(model_path)

    if task == 'mse':
        _, _, _, _, test_score, test_label, test_neg_score, test_neg_label = test_func(
            test_dataset, device, model, {}
        )

        test_score = torch.exp(test_score)
        test_neg_score = torch.exp(test_neg_score)
        combined_test_score = torch.cat((test_score, test_neg_score))

        combined_test_label = torch.cat((test_label, test_neg_label))

        score_np = combined_test_score.cpu().detach().numpy()
        label_np = combined_test_label.cpu().detach().numpy()

        df = pd.DataFrame({
            "score": score_np,
            "label": label_np
        })

        filename = f"quan_{data_name}_{seed}_{quantile}.csv"
        df.to_csv(filename, index=False)


if __name__ =="__main__":
    base_dir = './trained_models'

    for dataset in os.listdir(base_dir):
        if dataset == "nl27k":
            dataset_path = os.path.join(base_dir, dataset)
            if os.path.isdir(dataset_path):
                for model_folder in os.listdir(dataset_path):
                    model_folder_path = os.path.join(dataset_path, model_folder)
                    print(f"python test.py --model_path {model_folder_path}")
                    main(parse_args(), model_folder_path)
    print("Done")
