
from utils import get_new_model
import os

print(os.getcwd())

from param import *
from trainer import run_train
import numpy as np
import torch
import random
import wandb
from dataset import *
import datetime

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def sweep_train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        set_seed(config.seed)
        params = set_params(config.data, config.task, config.seed)

        train_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')
        train_test_dataset = UncertainTripleDataset(params.data_dir, 'train.tsv')  # obsolete, not used
        dev_dataset = UncertainTripleDataset(params.data_dir, 'val.tsv')
        test_dataset = UncertainTripleDataset(params.data_dir, 'test.tsv')


        print(params.whichmodel)
        print(params.early_stop)
        run = wandb_initialize(params)

        if not os.path.exists(params.model_dir):
            os.makedirs(params.model_dir)

        model = get_new_model(params)

        wandb.watch(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=params.LR)

        run_train(
            model, run, train_dataset, train_test_dataset, dev_dataset, test_dataset,
            optimizer, params
        )

        print('done')

def main():
    # Define sweep configuration
    sweep_config = {
        'method': 'grid',  # Choose 'random', 'grid', or 'bayes'
        'parameters': {
            'seed': {'values': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},  # Sweep seeds
            'data': {'values': ['nl27k']},      #, 'nl27k'
            'task': {'values': ['mse']}         #, 'ndcg'
        }
    }

    # Initialize sweep
    custom_text = "beurre"
    project_name = f"{custom_text}_{sweep_config['parameters']['data']['values'][0]}"
    sweep_id = wandb.sweep(sweep_config, project=project_name)
    wandb.agent(sweep_id, function=sweep_train)

if __name__ =="__main__":
    main()





