import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import types
import math        
class adaptive_decomp_Normalization(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.configs = configs
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_size
        self.patch_num = math.ceil(self.seq_len / self.patch_size)

    def ada_series_decomp(self, x, kernel_list):
        trend_x = []
        for decomp_kernel in kernel_list:
            decomp_layer = nn.AvgPool1d(kernel_size=decomp_kernel, stride=1, padding=0)
            front = x[:, 0:1, :].repeat(1, (decomp_kernel - 1) // 2, 1)
            end = x[:, -1:, :].repeat(1, decomp_kernel // 2, 1)
            pad_x = torch.cat([front, x, end], dim=1)
            trend_part = decomp_layer(pad_x.permute(0, 2, 1))
            trend_part = trend_part.permute(0, 2, 1)
            trend_x.append(trend_part)
        trend = sum(trend_x) / len(trend_x)
        season = x - trend
        return season, trend

    def normalize(self, x, period, period_num):
        B, L, M = x.shape
        normalized_season = []
        stat_for_pred = []
        LP = len(period)
        season, trend = self.ada_series_decomp(x, period)
        length = period * period_num - L
        for i in range(LP):  
            if length[i] != 0:
                padding_num = length[i] // L + 1
                padding_season = season.repeat(1, padding_num, 1)
                padding = padding_season[:, -length[i]:, :]
                season_input = torch.cat([season, padding], dim=1)
            else:
                season_input = season

            season_input = season_input.reshape(B, -1, period[i], M)
            season_mean = torch.mean(season_input, dim=-2, keepdim=True)
            season_std = torch.std(season_input, dim=-2, keepdim=True)
            norm_season = (season_input - season_mean) / (season_std + 1e-5)
            norm_season = rearrange(norm_season, 'b n p m -> (b m) n p')
            normalized_season.append(norm_season)
            stat_for_pred.append((season_mean, season_std))
        
        # 对趋势项进行正则化
        trend_mean = torch.mean(trend, dim=1, keepdim=True)
        trend_std = torch.std(trend, dim=1, keepdim=True)
        norm_trend = (trend - trend_mean) / (trend_std + 1e-5)
        stat_for_pred.append((trend_mean.unsqueeze(-2), trend_std.unsqueeze(-2)))
        
        norm_trend = norm_trend.squeeze(-1)
        self.patch_num = math.ceil(L / self.patch_size)
        if self.patch_num * self.patch_size - L != 0:
            padding_trend = norm_trend[:, -(self.patch_num * self.patch_size - L):]
            norm_trend = torch.cat([norm_trend, padding_trend], dim=1)
        norm_trend = rearrange(norm_trend, 'b (n p) -> b n p', p=self.patch_size)
        return normalized_season, norm_trend, stat_for_pred
    
    def de_normalize(self, norm_pred_list, stat_pred_list, period, period_weight, period_num):
        B, L, M = norm_pred_list[0].shape
        LP = len(period)
        denormalized_season = []
        trend_mean = stat_pred_list[LP][:, :, :M]
        trend_std = stat_pred_list[LP][:, :, M:]
        denorm_outputs = norm_pred_list[LP] * (trend_std + 1e-5) + trend_mean

        length = period * period_num - L
        for i in range(LP):               
            input = norm_pred_list[i]
            if length[i] != 0:
                padding_num = length[i] // L + 1
                padding_input = input.repeat(1, padding_num, 1)
                padding = padding_input[:, -length[i]:, :]
                input = torch.cat([input, padding], dim=1)
            input = input.reshape(B, -1, period[i], M)

            mean = stat_pred_list[i][:, :, :M].unsqueeze(2)
            std = stat_pred_list[i][:, :, M:].unsqueeze(2)
            denorm_output = input * (std + 1e-5) + mean
            denorm_output = denorm_output.reshape(B, -1, M)[:, :L, :]
            denormalized_season.append(denorm_output)
        denormalized_season = torch.stack(denormalized_season, dim=0)
        weight = torch.tensor(period_weight).view(-1, 1, 1, 1).to(denormalized_season.device)
        weighted_denorm_season = torch.sum(denormalized_season * weight, 0).squeeze(0)
        denorm_outputs = denorm_outputs + weighted_denorm_season
        return denorm_outputs



class PredModule(nn.Module):
    def __init__(self, configs, seq_len, pred_len):
        super(PredModule, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = 1
        self.model_mean = MLP(input_len=(configs.seq_len), 
                              seq_len=seq_len, 
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='mean')
        self.model_std = MLP(input_len=(configs.seq_len), 
                              seq_len=seq_len, 
                              pred_len=pred_len, 
                              enc_in=self.channels,
                              mode='std')
        self.weight = nn.Parameter(torch.ones(2, self.channels)) 
    
    def forward(self, mean, std):
        outputs_mean = self.model_mean(mean.squeeze(2))
        outputs_std = self.model_std(std.squeeze(2))
        outputs = torch.cat([outputs_mean, outputs_std], dim=-1)
        return outputs

class MLP(nn.Module):
    def __init__(self, input_len, seq_len, pred_len, enc_in, mode):
        super(MLP, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.input_len = input_len
        self.mode = mode
        if mode == 'std':
            self.final_activation = nn.ReLU()
        else:
            self.final_activation = nn.Identity()
        self.input = nn.Linear(self.seq_len, 512)
        self.input_raw = nn.Linear(self.input_len, 512)
        self.activation = nn.ReLU() if mode == 'std' else nn.Tanh()
        self.output = nn.Linear(512, self.pred_len)
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.input(x)
        x = self.output(self.activation(x))
        x = self.final_activation(x)
        return x.permute(0, 2, 1)
    

if __name__ == "__main__":
    num_points = 336
    period_ori = 24
    x = torch.linspace(0, 10 * period_ori, num_points)
    x_repeated = x.repeat(32, 1)
    device = torch.device('cuda:0')
    y = torch.cos(x_repeated)
    tensor = y.unsqueeze(2).to(device)
    configs = {
        "seq_len": 96,
        "pred_len": 96,
        "enc_in":1,
        "features":"S"
    }
    configs = types.SimpleNamespace(**configs)
    decomp = adaptive_decomp_Normalization(configs, device).cuda()
    season, trend = decomp.normalize(tensor)