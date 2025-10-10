from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate, del_files, visual, vali, test
from tqdm import tqdm
from models.PatchTST import PatchTST
from models.ARANSMTSllm import ARANSMTSllm
from models.DLinear import DLinear
import torch.nn.functional as F
from einops import rearrange
from models.GANModels import Generator, Discriminator
from layers.ada_dec_IN import adaptive_decomp_Normalization, PredModule
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs

import numpy as np
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

import argparse
import random

warnings.filterwarnings('ignore')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

parser = argparse.ArgumentParser(description='GPT4TS')

parser.add_argument('--model_id', type=str, required=True, default='test')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/')

parser.add_argument('--root_path', type=str, default='./dataset/traffic/')
parser.add_argument('--data_path', type=str, default='traffic.csv')
parser.add_argument('--data', type=str, default='custom')
parser.add_argument('--features', type=str, default='M')
parser.add_argument('--freq', type=int, default=1)
parser.add_argument('--target', type=str, default='OT')
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--percent', type=int, default=10)

parser.add_argument('--seq_len', type=int, default=512)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)

parser.add_argument('--decay_fac', type=float, default=0.75)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--num_workers', type=int, default=6)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--patience', type=int, default=2)

parser.add_argument('--gpt_layers', type=int, default=3)
parser.add_argument('--is_gpt', type=int, default=1)
parser.add_argument('--e_layers', type=int, default=3)
parser.add_argument('--d_model', type=int, default=768)
parser.add_argument('--n_heads', type=int, default=16)
parser.add_argument('--d_ff', type=int, default=512)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--enc_in', type=int, default=862)
parser.add_argument('--c_out', type=int, default=862)
parser.add_argument('--patch_size', type=int, default=16)
parser.add_argument('--kernel_size', type=int, default=25)

parser.add_argument('--loss_func', type=str, default='mse')
parser.add_argument('--pretrain', type=int, default=1)
parser.add_argument('--freeze', type=int, default=1)
parser.add_argument('--model', type=str, default='model')
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--max_len', type=int, default=-1)
parser.add_argument('--hid_dim', type=int, default=16)
parser.add_argument('--tmax', type=int, default=10)

parser.add_argument('--itr', type=int, default=3)
parser.add_argument('--cos', type=int, default=0)

args = parser.parse_args()
ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)


SEASONALITY_MAP = {
   "minutely": 1440,
   "10_minutes": 144,
   "half_hourly": 48,
   "hourly": 24,
   "daily": 7,
   "weekly": 1,
   "monthly": 12,
   "quarterly": 4,
   "yearly": 1
}

mses = []
maes = []

for ii in range(args.itr):

    setting = '{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_gl{}_df{}_eb{}_itr{}'.format(args.model_id, 512, args.label_len, args.pred_len,
                                                                    args.d_model, args.n_heads, args.e_layers, args.gpt_layers, 
                                                                    args.d_ff, args.embed, ii)
    path = os.path.join(args.checkpoints, setting)
    if not os.path.exists(path) and accelerator.is_local_main_process:
        os.makedirs(path)

    if args.freq == 0:
        args.freq = 'h'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_data, train_loader, RAG_Knowledge_base = data_provider(args, 'train')
    vali_data, vali_loader, _ = data_provider(args, 'val')
    test_data, test_loader, _ = data_provider(args, 'test')
    chunk_length = args.seq_len  + args.pred_len
    num_chunks = ((len(RAG_Knowledge_base) - chunk_length) + 1)
    chunks_x = []
    for i in range(num_chunks):
        feat_id = i // num_chunks
        start = i % num_chunks
        end = start + chunk_length
        chunk_x = RAG_Knowledge_base[start:end, feat_id]
        chunks_x.append(torch.tensor(chunk_x, dtype=torch.float, device=device))
    knowledge_base_all = torch.stack(chunks_x, dim=0)
    amps = 0.0
    count = 0
    for data in train_loader:
        lookback_window = data[0]
        b, l, dim = lookback_window.size()
        amps += (abs(torch.fft.rfft(lookback_window, dim=1))** 2).mean(dim=0).mean(dim=-1)
        count+=1
    amps = amps / count
    threshold = - amps.mean() * torch.log(torch.tensor(1e-4))
    amps[0] = 0
    top_list_ori = torch.nonzero(amps > threshold)
    top_list_ori = top_list_ori.squeeze(-1).detach().cpu().numpy()
    top_list = top_list_ori[top_list_ori != 1]
    if len(top_list) == 0:
        period = []
        period_weight = []
    else:
        period = args.seq_len // top_list
        period_power = amps[top_list]
        period_weight = period_power / period_power.sum()
        period_weight = period_weight.float()

    if args.freq != 'h':
        args.freq = SEASONALITY_MAP[test_data.freq]
        print("freq = {}".format(args.freq))
    time_now = time.time()
    train_steps = len(train_loader)

    model = ARANSMTSllm(args, device, period, period_weight, knowledge_base_all)

    params = model.parameters()
    model_optim = torch.optim.Adam(params, lr=args.learning_rate)
    early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience, verbose=True)
    criterion = nn.L1Loss()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=args.tmax, eta_min=1e-8)

    train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
        train_loader, vali_loader, test_loader, model, model_optim, scheduler)
    
    for epoch in range(args.train_epochs):

        iter_count = 0
        train_loss = []
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            iter_count += 1
            model_optim.zero_grad()
            batch_x = batch_x.to(torch.bfloat16).to(accelerator.device)
            batch_y = batch_y[:, -args.pred_len:, :].to(torch.bfloat16).to(accelerator.device)
    
            outputs, stat_pred, stat_of_y= model(batch_x, batch_y, ii)

            outputs = outputs[:, -args.pred_len:, :]
            loss = 0.8*criterion(outputs, batch_y) + 0.2*criterion(stat_pred, stat_of_y)
            if isinstance(loss, torch.Tensor) and loss.dim() > 0:
                loss = loss.mean()
            train_loss.append(loss.item())

            if (i + 1) % 1000 == 0:
                accelerator.print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            accelerator.backward(loss)
            model_optim.step()

        train_loss = np.average(train_loss)
        vali_loss = vali(accelerator, model, vali_data, vali_loader, criterion, args, device, ii)
        accelerator.print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
            epoch + 1, train_steps, train_loss, vali_loss))

        scheduler.step()
        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))

        early_stopping(vali_loss, model, path)
        if early_stopping.early_stop:
            accelerator.print("Early stopping")
            break

    best_model_path = path + '/' + 'checkpoint'
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    unwrapped_model.load_state_dict(torch.load(best_model_path, map_location=lambda storage, loc: storage))
    print("------------------------------------")
    mse, mae, rmse, smape = test(unwrapped_model, test_data, test_loader, args, device, ii)
    mses.append(mse)
    maes.append(mae)
