import numpy as np
from models.GANModels import Generator, Discriminator
from peft import LoraConfig, TaskType, get_peft_model
import torch.nn.functional as F
import torch
import math
import torch.nn as nn
from torch import optim
from layers.ada_dec_IN import adaptive_decomp_Normalization, PredModule
from layers.SelfAttention_Family import FullAttention, AttentionLayer, MultiHeadAttentionModel
from transformers import GPT2Model, GPT2Tokenizer, LlamaConfig, LlamaModel, LlamaTokenizer
from transformers import BertTokenizer, BertModel
from einops import rearrange
from embed import DataEmbedding, DataEmbedding_wo_time
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.GPT2_arch import AccustumGPT2Model
from models.Llama_arch import AccustumLlamaModel

class ARANSMTSllm(nn.Module):
    def __init__(self, configs, device, period_list, period_weight, knowledge_base_all):
        super(ARANSMTSllm, self).__init__()
        self.configs = configs
        self.device = device
        self.d_model = configs.d_model
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.is_gpt = configs.is_gpt
        self.pretrain = configs.pretrain
        self.batch_size = configs.batch_size

        self.ada_norm = adaptive_decomp_Normalization(configs)
        self.stat_predict = nn.ModuleList()
        self.in_layer = nn.ModuleList()
        self.out_layer = nn.ModuleList()
        self.retrieval_attention = nn.ModuleList()

        self.patch_size = configs.patch_size
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, 
            inference_mode=False, 
            r=8,
            lora_alpha=4,
            lora_dropout=0.3,
            target_modules=["q_proj", "k_proj"]
        )

        if configs.is_gpt:
            if configs.pretrain:
                self.gpt2 = AccustumLlamaModel.from_pretrained('', output_attentions=True, output_hidden_states=True, torch_dtype=torch.bfloat16)
                tokenizer = LlamaTokenizer.from_pretrained('', trust_remote_code=True, local_files_only=False)
                eos_token_id = tokenizer.eos_token_id
                embedding_matrix = self.gpt2.get_input_embeddings().weight
                self.eos_embedding = embedding_matrix[eos_token_id]
                self.eos_embedding = self.eos_embedding.detach().unsqueeze(0).repeat(self.batch_size, 1, 1).to(torch.bfloat16).to(self.device)
            else:
                print("------------------no pretrain------------------")
                self.gpt2 = LlamaTokenizer(LlamaConfig())
            self.gpt2 = get_peft_model(self.gpt2, peft_config)
        
        if configs.freeze and configs.pretrain:
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name or 'lora' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        self.topk = 1
        self.period_list = period_list
        self.period_weight = period_weight
        self.rag_patch_num = np.ceil((self.seq_len * (self.topk+1)) / self.period_list).astype(int)
        self.seq_patch_num = np.ceil(self.seq_len / self.period_list).astype(int)
        self.pred_patch_num = np.ceil(self.pred_len / self.period_list).astype(int)
        self.corr_pred_patch_num = np.ceil((self.pred_len * self.topk) / self.period_list).astype(int)
        self.seq_pred_patch_num = np.ceil((self.seq_len + self.pred_len) / self.period_list).astype(int)

        gan_checkpoint = torch.load("")
        self.gen_net = Generator(configs = configs, period_list=period_list)
        self.gen_net.load_state_dict(gan_checkpoint)
        for par in self.gen_net.parameters():
            par.requires_grad = False
        self.gen_net.eval().to(device=device)
        self.mask = []
        for i in range(len(period_list)):
            self.mask.append(torch.zeros(self.batch_size, self.pred_patch_num[i], self.d_model, dtype=torch.bfloat16).to(self.device))
            self.stat_predict.append(PredModule(configs=self.configs, seq_len=self.seq_patch_num[i], pred_len=self.pred_patch_num[i]).to(self.device))
            self.out_layer.append(nn.Linear((self.d_model * (self.seq_patch_num[i] + self.pred_patch_num[i])), (self.seq_len + self.pred_len)).to(self.device))
        self.patch_num_05 = math.ceil(self.seq_len / self.patch_size)  
        self.patch_num = math.ceil(self.seq_len * (self.topk+1) / self.patch_size)
        self.value_num = math.ceil(self.pred_len / self.patch_size)
        self.value_patch_num = np.ceil((self.seq_len + self.pred_len) / self.patch_size).astype(int)

        self.mask.append(torch.zeros(self.batch_size, self.value_num, self.d_model, dtype=torch.bfloat16).to(self.device))
        self.stat_predict.append(PredModule(configs=self.configs, seq_len=1, pred_len=1).to(self.device))
        self.out_layer.append(nn.Linear((self.d_model * (self.value_num + self.patch_num_05)), (self.seq_len + self.pred_len)).to(self.device))
        self.knowledge_base_all = knowledge_base_all
        self.dropout = nn.Dropout(0.3)

        for layer in (self.gpt2, self.stat_predict, self.out_layer):
            layer.to(device=device)
            layer.train()


    def retrieval_similarity(self, input):
        input_expand = input.unsqueeze(1)
        self.knowledge_base_all = self.knowledge_base_all.to(input.device)
        knowledge_base_expand = self.knowledge_base_all.unsqueeze(0)
        knowledge_history = knowledge_base_expand[:, :, :self.seq_len, :]
        differences = input_expand - knowledge_history
        distances = torch.norm(differences, dim=-2)
        distances = distances.squeeze(-1)
        _, topk_indices=torch.topk(-1*distances, self.topk)
        corr_ts = self.knowledge_base_all[topk_indices[:, 0]]
        return corr_ts

    def forward(self, x, y, itr):
        B, L, M = x.shape
        LP = len(self.period_list)
        corr_ts = self.retrieval_similarity(x)
        corr_season_list, corr_trend, _ = self.ada_norm.normalize(corr_ts, self.period_list, self.seq_pred_patch_num)
        corr_input = corr_season_list + [corr_trend]
        corr_ts_emb_list,_ = self.gen_net(corr_input)
        norm_season_list, norm_trend, stat_for_pred_x = self.ada_norm.normalize(x, self.period_list, self.seq_patch_num)
        norm_input = norm_season_list + [norm_trend]
        ts_emb_list,_ = self.gen_net(norm_input)

        pred = []
        stat_pred = []
        for i in range(LP):
            outputs = torch.cat([corr_ts_emb_list[i], self.eos_embedding, ts_emb_list[i], self.mask[i]], dim=1)
            if self.is_gpt:
                outputs = self.gpt2(inputs_embeds=outputs)#.last_hidden_state
                outputs = outputs[:, -(self.seq_patch_num[i] + self.pred_patch_num[i]):, :]
            outputs = self.dropout(self.out_layer[i](outputs.reshape(B*M, -1)))
            outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
            pred.append(outputs[:, -self.pred_len:, :])
            del outputs
            torch.cuda.empty_cache()

            season_part_mean, season_part_std = stat_for_pred_x[i]
            stat_pred.append(self.stat_predict[i](season_part_mean, season_part_std))

        trend_outputs = torch.cat([corr_ts_emb_list[LP], self.eos_embedding, ts_emb_list[LP], self.mask[LP]], dim=1)
        if self.is_gpt:
            trend_outputs = self.gpt2(inputs_embeds=trend_outputs)#.last_hidden_state
            trend_outputs = trend_outputs[:, -(self.value_num + self.patch_num_05):, :]
        trend_outputs = self.dropout(self.out_layer[LP](trend_outputs.reshape(B*M, -1)))
        trend_outputs = rearrange(trend_outputs, '(b m) l -> b l m', b=B)
        pred.append(trend_outputs[:, -self.pred_len:, :])

        trend_part_mean, trend_part_std = stat_for_pred_x[LP]
        stat_pred.append(self.stat_predict[LP](trend_part_mean, trend_part_std))


        outputs = self.ada_norm.de_normalize(pred, stat_pred, self.period_list, self.period_weight, self.pred_patch_num)
        _, _, stat_of_y = self.ada_norm.normalize(y, self.period_list, self.pred_patch_num)
        stat_pred = torch.cat(stat_pred, dim=1)

        mean_and_std_of_y = [torch.cat([item[0], item[1]], dim=-1).squeeze(-2) for item in stat_of_y]
        stat_y = torch.cat(mean_and_std_of_y, dim=1)
        
        return outputs, stat_pred, stat_y
