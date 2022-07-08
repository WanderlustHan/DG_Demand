import argparse
import pickle
import shutil, yaml, torch

import numpy as np
import torch.nn as nn
import os

from models.GMSDR import DynamicGraphNet
from models.model import create_model
from utils.train import train_model, test_model
from utils.util import get_optimizer, get_loss, get_scheduler
from utils.data_container import get_data_loader
from utils.preprocess import preprocessing_for_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument('--type', default='bike', type=str,
                        help='Type of dataset for training the model.')
parser.add_argument('--samples', default='data/nogrid/bike_samples.npz', type=str,
                    help='Path to the samples of the dataset.')
parser.add_argument('--graph_model', default='save/graph/nogrid_bike/best_graph_model.pkl', type=str,
                    help='Path to the graph model.')

args = parser.parse_args()

def _init_seed(SEED=10):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

def train(conf, data_category, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(conf['device'])
    device = torch.device(0)

    optimizer_name = conf['optimizer']['name']
    data_set = conf['data']['dataset']
    scheduler_name = conf['scheduler']['name']
    loss = get_loss(**conf['loss'])

    loss.to(device)

    eigen_mx = torch.from_numpy(np.load(args.samples)['embedding']).to(device)
    support = preprocessing_for_metric(data_category=data_category, dataset=conf['data']['dataset'],
                                       Normal_Method=conf['data']['Normal_Method'], _len=conf['data']['_len'],
                                       **conf['preprocess'])
    graph_model = DynamicGraphNet(**conf['model']['Dynamic'])
    save_dict = torch.load(args.graph_model)
    graph_model.load_state_dict(save_dict['model_state_dict'])
    graph_model.set_require_gard(False)

    model, trainer = create_model(loss,
                                  conf['model'],
                                  device,
                                  support)
    optimizer = get_optimizer(optimizer_name, model.parameters(), conf['optimizer'][optimizer_name]['lr'])
    scheduler = get_scheduler(scheduler_name, optimizer, **conf['scheduler'][scheduler_name])

    if torch.cuda.device_count() > 1:
        print("use ", torch.cuda.device_count(), "GPUS")
        model = nn.DataParallel(model)
    else:
        model.to(device)

    save_folder = os.path.join('save', conf['name'], f'{data_set}_{"".join(data_category)}', conf['tag'])
    run_folder = os.path.join('run', conf['name'], f'{data_set}_{"".join(data_category)}', conf['tag'])
    shutil.rmtree(save_folder, ignore_errors=True)
    os.makedirs(save_folder)
    shutil.rmtree(run_folder, ignore_errors=True)
    os.makedirs(run_folder)

    with open(os.path.join(save_folder, 'config.yaml'), 'w+') as _f:
        yaml.safe_dump(conf, _f)
        _f.close()

    data_loader, normal = get_data_loader(**conf['data'], data_category=data_category, device=device)

    train_model(model=model,
                graph_model=graph_model,
                dataloaders=data_loader,
                trainer=trainer,
                optimizer=optimizer,
                normal=normal,
                scheduler=scheduler,
                folder=save_folder,
                tensorboard_folder=run_folder,
                device=device,
                **conf['train'],
                i_embedding=eigen_mx)
    test_model(folder=save_folder,
               trainer=trainer,
               model=model,
               graph_model=graph_model,
               normal=normal,
               dataloaders=data_loader,
               conf=conf,
               device=device,
               i_embedding=eigen_mx)


if __name__ == '__main__':
    _init_seed(64)
    con = "config-" + args.type
    data = [args.type]
    with open(os.path.join('config', f'{con}.yaml')) as f:
        conf = yaml.safe_load(f)
    train(conf, data,args)
