import argparse
import datetime
import pickle
import shutil, yaml, torch
import torch.nn as nn
import numpy as np
import os

from models.GMSDR import DynamicGraphNet
from utils.train import train_graph_model
from utils.util import get_optimizer, get_loss, get_scheduler
from utils.data_container import get_data_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='taxi', type=str,
                    help='Type of dataset for training the model.')
parser.add_argument('--samples', default='data/nogrid/taxi_samples.npz', type=str,
                    help='Type of dataset for training the model.')
args = parser.parse_args()


def _init_seed(SEED=10):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


def generate_graph_train(conf, data_category, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf['device'])

    optimizer_name = conf['optimizer']['name']
    data_set = conf['data']['dataset']
    scheduler_name = conf['scheduler']['name']
    loss = get_loss(**conf['loss'])

    loss.to(device)

    eigen_mx = torch.from_numpy(np.load(args.samples)['embedding']).to(device)
    model = DynamicGraphNet(**conf['model']['Dynamic'])
    optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizer'][optimizer_name]['lr'])
    scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])

    if torch.cuda.device_count() > 1:
        print("use ", torch.cuda.device_count(), "GPUS")
        model = nn.DataParallel(model)
    else:
        model.to(device)

    save_folder = os.path.join('save', 'graph', f'{data_set}_{"".join(data_category)}')
    run_folder = os.path.join('run', 'graph', f'{data_set}_{"".join(data_category)}')
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)
    shutil.rmtree(run_folder, ignore_errors=True)
    os.makedirs(run_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(conf, _f)
        _f.close()

    data_loader, normal = get_data_loader(**conf['data'], data_category=data_category, device=device)

    train_graph_model(model=model,
                      dataloaders=data_loader,
                      optimizer=optimizer,
                      normal=normal,
                      scheduler=scheduler,
                      folder=save_folder,
                      tensorboard_folder=run_folder,
                      device=device,
                      **conf['train'],
                      i_embedding=eigen_mx,
                      sample_file=args.samples)


if __name__ == '__main__':
    _init_seed(64)
    con = "config-" + args.type
    data = [args.type]
    with open(os.path.join('config', f'{con}.yaml')) as f:
        conf = yaml.safe_load(f)
    generate_graph_train(conf, data, args)
