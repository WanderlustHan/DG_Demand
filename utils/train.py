import os
import numpy as np
import math
import json
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
import copy, time
from utils.util import save_model, get_number_of_parameters
from collections import defaultdict
from utils.evaluate import evaluate
from utils.util import MyEncoder
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_samples_emb(i_embedding, batch_it, batch_size, sample_file):
    samples_index = np.load(sample_file)
    # 选择当前batch下的正负样本index
    positive_index = samples_index['positive'][batch_it * batch_size:(batch_it + 1) * batch_size]
    negative_index = samples_index['negative'][batch_it * batch_size:(batch_it + 1) * batch_size]
    # 取每个Node对应的正负样本
    posi_emb = []
    nega_emb = []
    for i in range(batch_size):
        posi_emb.append(i_embedding[positive_index[i]])
        nega_emb.append(i_embedding[negative_index[i]])
    return torch.stack(posi_emb, dim=0).to(device), torch.stack(nega_emb, dim=0).to(device)


def prepare_data(x, y):
    """
    :param x: shape (batch_size, seq_len, num_sensor, input_dim)
    :param y: shape (batch_size, horizon, num_sensor, input_dim)
    :returns x: shape (batch_size, num_sensor, seq_len, input_dim)
             y: shape (batch_size, num_sensor, horizon, input_dim)
    """
    x = x.permute(0, 2, 1, 3)
    y = y.permute(0, 2, 1, 3)
    return x.to(device), y.to(device)

def generate_support(x,i_embedding,graph_model,device):
    embedding = i_embedding.repeat(x.shape[0], 1, 1).to(device)  # [batch_size, num_sensor, emb_dim]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, num_sensor, seq_len*input_dim)
    input = torch.cat((embedding, x), dim=2)  # embedding + 2*12 time
    output = graph_model(input)  # output is new embedding [batch,node,emb]
    supports = torch.cosine_similarity(output.unsqueeze(2), output.unsqueeze(1), dim=-1)
    adj_index = supports.topk(30, dim=-1).indices  # [batch,node,topk]
    sup = torch.zeros(supports.shape).to(device)
    supports = sup.scatter(2, adj_index, 0.005) # [batch,node,node]
    return supports

def train_model(model: nn.Module,
                graph_model: nn.Module,
                dataloaders,
                optimizer,
                normal,
                scheduler,
                folder: str,
                trainer,
                tensorboard_folder,
                epochs: int,
                device,
                max_grad_norm: float = None,
                early_stop_steps: float = None,
                i_embedding=None):
    save_path = os.path.join(folder, 'best_model.pkl')
    if os.path.exists(save_path):
        print("path exist")
        save_dict = torch.load(save_path)
        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        print("path does not exist")
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate', 'test']
    writer = SummaryWriter(tensorboard_folder)
    since = time.perf_counter()
    model = model.to(device)
    graph_model = graph_model.to(device)
    print(f'Trainable parameters: {get_number_of_parameters(model) + get_number_of_parameters(graph_model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                    graph_model.train()
                else:
                    model.eval()
                    graph_model.eval()
                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    running_targets.append(targets.numpy())
                    with torch.no_grad():
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        x, y = prepare_data(inputs, targets)
                        supports = generate_support(x,i_embedding,graph_model,device)
                        # 以下都是完整训练模型的代码
                        outputs, loss = trainer.train(inputs, targets, supports, phase)
                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(outputs.cpu().numpy())
                    running_loss[phase] += loss.item() * len(targets)
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {normal[0].rmse_transform(running_loss[phase] / steps):3.6}')
                    torch.cuda.empty_cache()
                running_metrics[phase] = evaluate(np.concatenate(predictions), np.concatenate(running_targets), normal)
                running_metrics[phase].pop('rmse')
                running_metrics[phase].pop('pcc')
                running_metrics[phase].pop('mae')
                if phase == 'validate':
                    model.eval()
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                        process_test(folder, trainer, model, graph_model,normal, dataloaders, device, epoch,i_embedding)
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')
            scheduler.step(running_loss['train'])

            for metric in running_metrics['train'].keys():
                for phase in phases:
                    for key, val in running_metrics[phase][metric].items():
                        writer.add_scalars(f'{metric}/{key}', {f'{phase}': val}, global_step=epoch)
            writer.add_scalars('Loss', {
                f'{phase} loss': running_loss[phase] / len(dataloaders[phase].dataset) for phase in phases},
                               global_step=epoch)
    except (ValueError, KeyboardInterrupt):
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')


def process_test(folder: str,
                 trainer,
                 model,
                 graph_model,
                 normal,
                 dataloaders,
                 device,
                 epoch,
                 i_embedding):
    save_path = os.path.join(folder, 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())
        x, y = prepare_data(inputs, targets)
        supports = generate_support(x,i_embedding,graph_model,device)
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, loss = trainer.train(inputs, targets, supports, 'test')
            predictions.append(outputs.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)
    scores = evaluate(running_targets, predictions, normal)
    print('test results in epoch of ' + str(epoch) + ':')
    print("\t rmse: " + json.dumps(scores['rmse']) + ",")
    print("\t mae: " + json.dumps(scores['mae']) + ",")
    print("\t pcc: " + json.dumps(scores['pcc']) + ",")


def test_model(folder: str,
               trainer,
               model,
               graph_model,
               normal,
               dataloaders,
               conf,
               device,
               i_embedding):
    save_path = os.path.join(folder, 'best_model.pkl')
    save_dict = torch.load(save_path)
    model.load_state_dict(save_dict['model_state_dict'])
    model.eval()
    steps, predictions, running_targets = 0, list(), list()
    tqdm_loader = tqdm(enumerate(dataloaders['test']))
    for step, (inputs, targets) in tqdm_loader:
        running_targets.append(targets.numpy())
        x, y = prepare_data(inputs, targets)
        supports = generate_support(x,i_embedding,graph_model,device)
        with torch.no_grad():
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, loss = trainer.train(inputs, targets, supports, 'test')
            predictions.append(outputs.cpu().numpy())

    running_targets, predictions = np.concatenate(running_targets, axis=0), np.concatenate(predictions, axis=0)

    scores = evaluate(running_targets, predictions, normal)
    print('test results:')
    print(json.dumps(scores, cls=MyEncoder, indent=4))
    np.savez(os.path.join(folder, 'test-results.npz'), predictions=predictions, targets=running_targets)


def train_graph_model(model: nn.Module,
                dataloaders,
                optimizer,
                normal,
                scheduler,
                folder: str,
                tensorboard_folder,
                epochs: int,
                device,
                max_grad_norm: float = None,
                early_stop_steps: float = None,
                i_embedding=None,
                sample_file: str = None):
    save_path = os.path.join(folder, 'best_graph_model.pkl')
    if os.path.exists(save_path):
        print("path exist")
        save_dict = torch.load(save_path)
        model.load_state_dict(save_dict['model_state_dict'])
        optimizer.load_state_dict(save_dict['optimizer_state_dict'])
        best_val_loss = save_dict['best_val_loss']
        begin_epoch = save_dict['epoch'] + 1
    else:
        print("path does not exist")
        save_dict = dict()
        best_val_loss = float('inf')
        begin_epoch = 0

    phases = ['train', 'validate']
    writer = SummaryWriter(tensorboard_folder)
    since = time.perf_counter()
    model = model.to(device)
    print(f'Trainable parameters: {get_number_of_parameters(model)}.')

    try:
        for epoch in range(begin_epoch, begin_epoch + epochs):
            running_loss, running_metrics = defaultdict(float), dict()
            for phase in phases:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()
                steps, predictions, running_targets = 0, list(), list()
                tqdm_loader = tqdm(enumerate(dataloaders[phase]))
                for step, (inputs, targets) in tqdm_loader:
                    # running_targets.append(targets.numpy())
                    with torch.no_grad():
                        inputs = inputs.to(device)
                        targets = targets.to(device)
                    with torch.set_grad_enabled(phase == 'train'):
                        x, y = prepare_data(inputs, targets)
                        embedding = i_embedding.repeat(x.shape[0], 1, 1).to(device)  # [batch_size, num_sensor, emb_dim]
                        x = x.reshape(x.shape[0], x.shape[1], -1)  # (batch_size, num_sensor, seq_len*input_dim)
                        input = torch.cat((embedding, x), dim=2)  # embedding + 2*12 time
                        output = model(input)  # output is new embedding [batch,node,emb]
                        posi_emb, nega_emb = _get_samples_emb(i_embedding, step, x.shape[0],sample_file)  # [batch,node,emb]
                        # 计算正负样本评分
                        pos_score = (output * posi_emb).sum(-1)
                        neg_score = (output * nega_emb).sum(-1)
                        loss = -1 * nn.LogSigmoid()((pos_score - neg_score)).mean()

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            if max_grad_norm is not None:
                                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                            optimizer.step()

                    with torch.no_grad():
                        predictions.append(output.cpu().numpy())
                    running_loss[phase] += loss.item()
                    steps += len(targets)

                    tqdm_loader.set_description(
                        f'{phase:5} epoch: {epoch:3}, {phase:5} loss: {normal[0].rmse_transform(running_loss[phase] / steps):3.6}')
                    torch.cuda.empty_cache()

                if phase == 'validate':
                    model.eval()
                    if running_loss['validate'] <= best_val_loss or math.isnan(running_loss['validate']):
                        best_val_loss = running_loss['validate']
                        save_dict.update(model_state_dict=copy.deepcopy(model.state_dict()),
                                         epoch=epoch,
                                         best_val_loss=best_val_loss,
                                         optimizer_state_dict=copy.deepcopy(optimizer.state_dict()))
                        save_model(save_path, **save_dict)
                        print(f'Better model at epoch {epoch} recorded.')
                        print(f'Best model of graph saved at {save_path}.')
                    elif epoch - save_dict['epoch'] > early_stop_steps:
                        raise ValueError('Early stopped.')
            scheduler.step(running_loss['train'])
    except (ValueError, KeyboardInterrupt):
        writer.close()
        time_elapsed = time.perf_counter() - since
        print(f"cost {time_elapsed} seconds")
        print(f'model of epoch {save_dict["epoch"]} successfully saved at `{save_path}`')
