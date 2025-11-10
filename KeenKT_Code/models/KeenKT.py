import random
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SEBlock(nn.Module):
    """
    通道重标定：输入 [B, T, H]，输出同形状
    reduction 默认为 16，可通过 --se_ratio 调整
    """
    def __init__(self, hidden_dim: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim // reduction, hidden_dim, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=1)                  # [B, H]  (Squeeze)
        s = self.fc2(self.relu(self.fc1(s))).unsqueeze(1)  # [B,1,H]
        s = self.sigmoid(s)                # Gate
        return x * s                       # [B,T,H]  (Excitation)





def nig_distance_matmul(mean1, cov1, mean2, cov2):
    """Compute pairwise distance between two sets of distributions (Normal-Inverse Gaussian approximated by mean and sqrt-var)."""
    # Mean part (squared Euclidean distance between mean vectors)
    mean1_sq = torch.sum(mean1 ** 2, dim=-1, keepdim=True)
    mean2_sq = torch.sum(mean2 ** 2, dim=-1, keepdim=True)
    mean_diff = mean1_sq + mean2_sq.transpose(-2, -1) - 2 * torch.matmul(mean1, mean2.transpose(-2, -1))
    # Covariance part (squared difference between sqrt-variances)
    cov1_sq = torch.sum(cov1 ** 2, dim=-1, keepdim=True)
    cov2_sq = torch.sum(cov2 ** 2, dim=-1, keepdim=True)
    cov_diff = cov1_sq + cov2_sq.transpose(-2, -1) - 2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-2, -1))
    return mean_diff + cov_diff

def d2s_1overx(distance):
    return 1 / (1 + distance)
    

class CosinePositionalEmbedding(nn.Module):
    """
    简单示例：用 sin / cos 函数构造的位置编码，常见于Transformer。
    这里叫 “CosinePositionalEmbedding” 只是个命名示例，也包含了 sin 分量。
    你也可以根据需求改成纯 cos 或其他自定义函数。
    """
    def __init__(self, d_model, max_len=5000):
        super(CosinePositionalEmbedding, self).__init__()
        # 创建一个 [max_len, d_model] 的位置编码矩阵 pe
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # shape=[max_len,1]
        # 计算 dim=2 的 div_term: 频率递增，常规Transformer做法
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        # 偶数维用 sin，奇数维用 cos
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # shape: [1, max_len, d_model]
        pe = pe.unsqueeze(0)
        # 注册为buffer，意味着在模型保存/加载时自动处理，但不会成为可训练参数
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: [batch_size, seq_len, d_model] 或 [batch_size, seq_len]
        这里仅用 seq_len 来截取相应长度的编码并返回。
        """
        seq_len = x.size(1)
        # 注意 self.pe[:, :seq_len, :] 只能取到序列长度之内的位置编码
        # shape=[1, seq_len, d_model]
        return self.pe[:, :seq_len, :]

#############################################
# 新增：扩散模块，用于对 latent state 进行噪声还原
#############################################
class DiffusionModule(nn.Module):
    def __init__(self, latent_dim):
        super(DiffusionModule, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

    def forward(self, x):
        # 输入 x 加入噪声后，输出去噪结果
        res = self.net(x)
        return x + res

#############################################
# Normal Inverse Gaussian (NIG) Contrastive Loss (替换高斯分布的 WassersteinNCE 损失)
#############################################
class NIGNCELoss(nn.Module):
    def __init__(self, temperature):
        super(NIGNCELoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.temperature = temperature
        self.activation = nn.ELU()

    def forward(self, mean1, cov1, mean2, cov2):
        # 将协方差表示（实际上为标准差）通过 ELU 激活确保为正值
        cov1 = self.activation(cov1) + 1
        cov2 = self.activation(cov2) + 1
        # 计算正样本和负样本的相似度矩阵（基于 NIG 分布距离）
        sim11 = d2s_1overx(nig_distance_matmul(mean1, cov1, mean1, cov1)) / self.temperature
        sim22 = d2s_1overx(nig_distance_matmul(mean2, cov2, mean2, cov2)) / self.temperature
        sim12 = -d2s_1overx(nig_distance_matmul(mean1, cov1, mean2, cov2)) / self.temperature
        d = sim12.shape[-1]
        # 将自身的匹配相似度设为 -inf，避免模型将同一分布视为正样本（不计入对比）
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        # 拼接 logits 并构造标签，进行 InfoNCE 对比损失计算
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        logits = torch.cat([raw_scores1, raw_scores2], dim=-2)
        labels = torch.arange(2 * d, dtype=torch.long, device=logits.device)
        loss = self.criterion(logits, labels)
        return loss

class Architecture(nn.Module):
    def __init__(self, n_question, n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type

        self.position_mean_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)
        self.position_cov_embeddings = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

        if model_type in {'KeenKT'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])

    def forward(self, q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data, atten_type='w2'):
        seqlen, batch_size = q_mean_embed_data.size(1), q_mean_embed_data.size(0)
        # 加入位置嵌入
        mean_q_posemb = self.position_mean_embeddings(q_mean_embed_data)
        cov_q_posemb = self.position_cov_embeddings(q_cov_embed_data)
        q_mean_embed_data = q_mean_embed_data + mean_q_posemb
        q_cov_embed_data = q_cov_embed_data + cov_q_posemb

        qa_mean_posemb = self.position_mean_embeddings(qa_mean_embed_data)
        qa_cov_posemb = self.position_cov_embeddings(qa_cov_embed_data)
        qa_mean_embed_data = qa_mean_embed_data + qa_mean_posemb
        qa_cov_embed_data = qa_cov_embed_data + qa_cov_posemb

        # 确保协方差参数非负
        elu_act = torch.nn.ELU()
        q_cov_embed_data = elu_act(q_cov_embed_data) + 1
        qa_cov_embed_data = elu_act(qa_cov_embed_data) + 1

        # 准备自注意力的 query/key (x) 和 value (y)
        y_mean = qa_mean_embed_data
        y_cov  = qa_cov_embed_data
        x_mean = q_mean_embed_data
        x_cov = q_cov_embed_data

        # 通过多层注意力块更新序列表示
        for block in self.blocks_2:
            x_mean, x_cov = block(mask=0, query_mean=x_mean, query_cov=x_cov,
                                  key_mean=x_mean, key_cov=x_cov,
                                  values_mean=y_mean, values_cov=y_cov,
                                  atten_type=atten_type, apply_pos=True)
        return x_mean, x_cov

class KEENKT(nn.Module):
    def __init__(self, n_question, n_pid, emb_type,  
                 use_uncertainty_aug, n_blocks, dropout, d_model, d_ff,
                 final_fc_dim, final_fc_dim2, seq_len, model_type='KeenKT', use_diffusion=True, diffusion_weight=0.1, noise_level=0.2,use_CL=True, cl_weight=1.0, num_attn_heads=8, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4,
            kq_same=1, atten_type='w2', emb_path="", pretrain_dim=768,separate_qa=False, n_heads=8, se_ratio: int = 16):
        super(KEENKT, self).__init__()
        self.model_name = "KEENKT"
        self.n_question = n_question
        self.n_pid = n_pid
        self.emb_type = emb_type
        self.separate_qa = separate_qa
        self.use_CL = use_CL
        self.cl_weight = cl_weight  
        self.use_uncertainty_aug = use_uncertainty_aug
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.d_model = d_model
        self.model_type = model_type
        self.kq_same = 1  # query and key use same projection
        if self.n_pid > 0 and "norasch" not in emb_type:
            # 定义题目差异嵌入
            self.q_embed_diff = nn.Embedding(n_question+1, d_model)
            
            # 可能还需要定义 combined QA diff
            self.qa_embed_diff = nn.Embedding(2, d_model)
            
            # 定义题目难度的嵌入
            self.difficult_param = nn.Embedding(n_pid+1, d_model)
        self.atten_type = 'w2' 

        embed_l = d_model

        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            # Embedding层：使用正态逆高斯（NIG）分布参数 (mu, alpha, beta, delta)
            # Embeddings for NIG parameters (mu, alpha, beta, delta) per question
            self.mu_q_embed = nn.Embedding(self.n_question, embed_l)
            self.alpha_q_embed = nn.Embedding(self.n_question, embed_l)
            self.beta_q_embed = nn.Embedding(self.n_question, embed_l)
            self.delta_q_embed = nn.Embedding(self.n_question, embed_l)

            if self.separate_qa:
                # Embeddings for combined question+answer ID (for separate_qa case)
                self.mu_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
                self.alpha_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
                self.beta_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
                self.delta_qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else:
                # Embeddings for answer correctness (for non-separate_qa case)
                self.mu_qa_embed = nn.Embedding(2, embed_l)
                self.alpha_qa_embed = nn.Embedding(2, embed_l)
                self.beta_qa_embed = nn.Embedding(2, embed_l)
                self.delta_qa_embed = nn.Embedding(2, embed_l)

        # Architecture 对象，包含多层注意力块
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=n_heads, dropout=dropout,
                                  d_model=d_model, d_feature=d_model / n_heads, d_ff=d_ff,
                                  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)
        
        self.se_gate = SEBlock(d_model, reduction=se_ratio)

        # 输出层：将四部分拼接后经过全连接得到最终预测
        self.out = nn.Sequential(
            nn.Linear(embed_l * 4, final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )
        self.reset()

        #############################################
        # 扩散模块相关参数初始化
        #############################################
        self.use_diffusion = use_diffusion
        self.diffusion_weight = diffusion_weight
        self.noise_level = noise_level
        if self.use_diffusion:
            self.diffusion_module = DiffusionModule(d_model)

        if use_CL:
            self.wloss = NIGNCELoss(1)
            self.cl_weight = cl_weight

    def reset(self):
        # 参数初始化函数
        for p in self.parameters():
            if hasattr(self, 'n_pid') and self.n_pid > 0 and p.size(0) == self.n_pid+1:
                # 针对 difficulty 参数进行初始化
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        """Get embeddings for question data and target (response) and convert to NIG distribution parameters."""
        # 获取题目和作答结果的嵌入向量
        q_mean_embed = self.mu_q_embed(q_data)
        q_alpha_embed = self.alpha_q_embed(q_data)
        q_beta_embed = self.beta_q_embed(q_data)
        q_delta_embed = self.delta_q_embed(q_data)
        if self.separate_qa:
            qa_data = q_data + self.n_question * target
            qa_mean_embed = self.mu_qa_embed(qa_data)
            qa_alpha_embed = self.alpha_qa_embed(qa_data)
            qa_beta_embed = self.beta_qa_embed(qa_data)
            qa_delta_embed = self.delta_qa_embed(qa_data)
        else:
            qa_mean_embed = self.mu_qa_embed(target) + q_mean_embed
            qa_alpha_embed = self.alpha_qa_embed(target) + q_alpha_embed
            qa_beta_embed = self.beta_qa_embed(target) + q_beta_embed
            qa_delta_embed = self.delta_qa_embed(target) + q_delta_embed
        # 将嵌入表示转换为实际的 NIG 分布参数 (计算实际均值和标准差)
        q_alpha_pos = F.softplus(q_alpha_embed) + 1e-8
        q_beta_con = torch.tanh(q_beta_embed) * q_alpha_pos * 0.999
        q_delta_pos = F.elu(q_delta_embed) + 1
        q_gamma = torch.sqrt(torch.clamp(q_alpha_pos**2 - q_beta_con**2, min=1e-8))
        q_mean_actual = q_mean_embed + (q_delta_pos * q_beta_con / torch.clamp(q_gamma, min=1e-8))
        q_sqrt_var = torch.sqrt(q_delta_pos) * q_alpha_pos / torch.clamp(q_gamma, min=1e-8)**1.5

        qa_alpha_pos = F.softplus(qa_alpha_embed) + 1e-8
        qa_beta_con = torch.tanh(qa_beta_embed) * qa_alpha_pos * 0.999
        qa_delta_pos = F.elu(qa_delta_embed) + 1
        qa_gamma = torch.sqrt(torch.clamp(qa_alpha_pos**2 - qa_beta_con**2, min=1e-8))
        qa_mean_actual = qa_mean_embed + (qa_delta_pos * qa_beta_con / torch.clamp(qa_gamma, min=1e-8))
        qa_sqrt_var = torch.sqrt(qa_delta_pos) * qa_alpha_pos / torch.clamp(qa_gamma, min=1e-8)**1.5

        return q_mean_actual, q_sqrt_var, qa_mean_actual, qa_sqrt_var

    def forward(self, dcur, qtest=False, train=False):
        # 从输入数据字典中获取序列
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:, 0:1], qshft), dim=1)
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)
        mask = dcur["masks"]

        # 对比学习时的数据增强（若使用的话）
        if train and self.use_CL:
            if self.use_uncertainty_aug:
                rshft_aug = dcur["shft_r_aug"].long()
                r_aug     = dcur["r_aug"].long()
                target_aug = torch.cat((r_aug[:, 0:1], rshft_aug), dim=1)
            else:
                target_aug = target

        emb_type = self.emb_type

        if emb_type.startswith("qid") or emb_type.startswith("stoc"):
            # 获取基础嵌入（题目分布和作答影响），并得到 NIG 分布的均值和标准差表示
            q_mean_embed_data, q_cov_embed_data, qa_mean_embed_data, qa_cov_embed_data = self.base_emb(q_data, target)
            if train and self.use_CL:
                mean_q_aug_embed_data, cov_q_aug_embed_data, mean_qa_aug_embed_data, cov_qa_aug_embed_data = self.base_emb(q_data, target_aug)

        # 如有题目难度参数且未使用 Rasch 模型，则将题目难度影响添加到嵌入中
        if self.n_pid > 0 and emb_type.find("norasch") == -1:
            if emb_type.find("aktrasch") == -1:
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data
                if train and self.use_CL:
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data * q_embed_diff_data
            else:
                q_embed_diff_data = self.q_embed_diff(q_data)
                pid_embed_data = self.difficult_param(pid_data)
                q_mean_embed_data = q_mean_embed_data + pid_embed_data * q_embed_diff_data
                q_cov_embed_data = q_cov_embed_data + pid_embed_data * q_embed_diff_data
                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_mean_embed_data = qa_mean_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
                qa_cov_embed_data = qa_cov_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)
                if train and self.use_CL:
                    qa_aug_embed_diff_data = self.qa_embed_diff(target_aug)
                    mean_q_aug_embed_data = mean_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    cov_q_aug_embed_data = cov_q_aug_embed_data + pid_embed_data * q_embed_diff_data
                    mean_qa_aug_embed_data = mean_qa_aug_embed_data + pid_embed_data * (qa_aug_embed_diff_data + q_embed_diff_data)
                    cov_qa_aug_embed_data = cov_qa_aug_embed_data + pid_embed_data * (qa_aug_embed_diff_data + q_embed_diff_data)

        # 通过 Architecture 模型得到最终的知识状态分布表征
        if emb_type in ["qid", "stoc_qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:
            mean_d_output, cov_d_output = self.model(q_mean_embed_data, q_cov_embed_data,
                                                    qa_mean_embed_data, qa_cov_embed_data,
                                                    self.atten_type)
            mean_d_output = self.se_gate(mean_d_output)
            cov_d_output  = self.se_gate(cov_d_output)

        # 若有对比学习，增强分支也需门控
            if train and self.use_CL:
                mean_d2_output, cov_d2_output = self.model(
                    mean_q_aug_embed_data, cov_q_aug_embed_data,
                    mean_qa_aug_embed_data, cov_qa_aug_embed_data,
                    self.atten_type
                )
                mean_d2_output = self.se_gate(mean_d2_output)
                cov_d2_output  = self.se_gate(cov_d2_output)

            # 扩散模块：训练时对 latent state 添加噪声并计算去噪重构误差
            if train and self.use_diffusion:
                latent = mean_d_output
                latent_noisy = latent + torch.randn_like(latent) * self.noise_level
                latent_denoised = self.diffusion_module(latent_noisy)
                diffusion_loss = F.mse_loss(latent_denoised, latent)
            else:
                diffusion_loss = 0.0

            # 对比学习：计算增强样本通过模型得到的表征
            if train and self.use_CL:
                mean_d2_output, cov_d2_output = self.model(mean_q_aug_embed_data, cov_q_aug_embed_data,
                                                          mean_qa_aug_embed_data, cov_qa_aug_embed_data,
                                                          self.atten_type)
                mas = mask
                true_tensor = torch.ones(mas.size(0), 1, dtype=torch.bool).to(device)
                mas = torch.cat((true_tensor, mas), dim=1).unsqueeze(-1)

                # 按掩码对序列取平均，得到每个序列的整体表示
                pooled_mean_d_output = torch.mean(mean_d_output * mas, dim=1)
                pooled_cov_d_output = torch.mean(cov_d_output * mas, dim=1)
                pooled_mean_d2_output = torch.mean(mean_d2_output * mas, dim=1)
                pooled_cov_d2_output = torch.mean(cov_d2_output * mas, dim=1)

                # 计算对比损失
                if emb_type == "stoc_qid":
                    loss = self.wloss(pooled_mean_d_output, pooled_cov_d_output,
                                      pooled_mean_d2_output, pooled_cov_d2_output)
                else:
                    # 非随机嵌入情况下，不考虑不确定性（cov 流），仅使用 mean 流进行对比
                    loss = self.wloss(pooled_mean_d_output, pooled_mean_d_output,
                                      pooled_mean_d2_output, pooled_mean_d2_output)

            # 计算协方差流的平均（用于输出温度或调试）
            activation = nn.ELU()
            temp = torch.mean(torch.mean(activation(cov_d_output) + 1, dim=-1), -1)

            # 拼接 mean 与 cov 流的输出，以及对应的初始嵌入，用于最终预测
            if emb_type == "stoc_qid":
                concat_q = torch.cat([mean_d_output, cov_d_output, q_mean_embed_data, q_cov_embed_data], dim=-1)
            else:
                concat_q = torch.cat([mean_d_output, mean_d_output, q_cov_embed_data, q_cov_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            preds = torch.sigmoid(output)

        # 根据训练或测试状态返回结果
        if train:
            if self.use_CL:
                total_loss = self.cl_weight * loss + self.diffusion_weight * diffusion_loss
                return preds, total_loss, temp
            else:
                return preds, self.diffusion_weight * diffusion_loss, 0
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

#############################################
# Attention机制：支持 NIG 分布的计算
#############################################
def attention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    # 标准点积注意力（均值流与方差流分别计算），并根据不确定性差异进行调整
    scores_mean = torch.matmul(q_mean, k_mean.transpose(-2, -1)) / math.sqrt(d_k)
    scores_cov = torch.matmul(q_cov, k_cov.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores_mean.size(0), scores_mean.size(1), scores_mean.size(2)
    x1 = torch.arange(seqlen).expand(seqlen, -1).to(device)
    x2 = x1.transpose(0, 1).contiguous()
    with torch.no_grad():
        scores_mean_ = scores_mean.masked_fill(mask == 0, -1e32)
        scores_cov_ = scores_cov.masked_fill(mask == 0, -1e32)
        scores_mean_ = F.softmax(scores_mean_, dim=-1)
        scores_cov_ = F.softmax(scores_cov_, dim=-1)
        scores_mean_ = scores_mean_ * mask.float().to(device)
        scores_cov_ = scores_cov_ * mask.float().to(device)
        distcum_scores_mean = torch.cumsum(scores_mean_, dim=-1)
        distcum_scores_cov = torch.cumsum(scores_cov_, dim=-1)
        disttotal_scores_mean = torch.sum(scores_mean_, dim=-1, keepdim=True)
        disttotal_scores_cov = torch.sum(scores_cov_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].float().to(device)
        dist_scores_mean = torch.clamp((disttotal_scores_mean - distcum_scores_mean) * position_effect, min=0.)
        dist_scores_cov = torch.clamp((disttotal_scores_cov - distcum_scores_cov) * position_effect, min=0.)
        dist_scores_mean = dist_scores_mean.sqrt().detach()
        dist_scores_cov = dist_scores_cov.sqrt().detach()
    m = nn.Softplus()
    gamma_val = -1. * m(gamma).unsqueeze(0)
    total_effect_mean = torch.clamp(torch.clamp((dist_scores_mean * gamma_val).exp(), min=1e-5), max=1e5)
    total_effect_cov = torch.clamp(torch.clamp((dist_scores_cov * gamma_val).exp(), min=1e-5), max=1e5)
    scores_mean = scores_mean * total_effect_mean
    scores_cov = scores_cov * total_effect_cov
    scores_mean.masked_fill_(mask == 0, -1e32)
    scores_cov.masked_fill_(mask == 0, -1e32)
    scores_mean = F.softmax(scores_mean, dim=-1)
    scores_cov = F.softmax(scores_cov, dim=-1)
    output_mean = torch.matmul(scores_mean, v_mean)
    output_cov = torch.matmul(scores_cov, v_cov)
    return output_mean, output_cov

def uattention(q_mean, q_cov, k_mean, k_cov, v_mean, v_cov, d_k, mask, dropout, zero_pad, gamma):
    # 使用分布距离（NIG 分布的距离）计算注意力权重（统一对均值和方差流）
    scores = - nig_distance_matmul(q_mean, q_cov, k_mean, k_cov) / math.sqrt(d_k)
    scores = scores.masked_fill(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    scores = dropout(scores)
    output_mean = torch.matmul(scores, v_mean)
    output_cov = torch.matmul(scores, v_cov)
    return output_mean, output_cov

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(d_model, d_feature, n_heads, dropout, kq_same=kq_same)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.mean_linear1 = nn.Linear(d_model, d_ff)
        self.cov_linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.mean_linear2 = nn.Linear(d_ff, d_model)
        self.cov_linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation2 = nn.ELU()

    def forward(self, mask, query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, atten_type='w2', apply_pos=True):
        seqlen, batch_size = query_mean.size(1), query_mean.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(device)

        if mask == 0:
            # 使用多头注意力（可选 w2 或 dp 模式）
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov,
                mask=src_mask, atten_type=atten_type, zero_pad=True
            )
        else:
            query2_mean, query2_cov = self.masked_attn_head(
                query_mean, query_cov, key_mean, key_cov, values_mean, values_cov,
                mask=src_mask, atten_type=atten_type, zero_pad=False
            )

        # 残差连接 + LayerNorm
        query_mean = query_mean + self.dropout1(query2_mean)
        query_cov = query_cov + self.dropout1(query2_cov)
        query_mean = self.layer_norm1(query_mean)
        # 对协方差流输出先激活再加1（确保正值），再 LayerNorm
        query_cov = self.layer_norm1(self.activation2(query_cov) + 1)

        if apply_pos:
            # 前向反馈网络（FFN）分别作用于 mean 和 cov 流
            query2_mean = self.mean_linear2(self.dropout(self.activation(self.mean_linear1(query_mean))))
            query2_cov = self.cov_linear2(self.dropout(self.activation(self.cov_linear1(query_cov))))
            query_mean = query_mean + self.dropout2(query2_mean)
            query_cov = query_cov + self.dropout2(query2_cov)
            query_mean = self.layer_norm2(query_mean)
            query_cov = self.layer_norm2(self.activation2(query_cov) + 1)
        return query_mean, query_cov

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.activation = nn.ELU()
        self.v_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.v_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_mean_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_mean_linear = nn.Linear(d_model, d_model, bias=bias)
            self.q_cov_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_mean_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_cov_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化多头注意力线性层参数
        xavier_uniform_(self.v_mean_linear.weight)
        xavier_uniform_(self.v_cov_linear.weight)
        xavier_uniform_(self.k_mean_linear.weight)
        xavier_uniform_(self.k_cov_linear.weight)
        if hasattr(self, "q_mean_linear"):
            xavier_uniform_(self.q_mean_linear.weight)
            xavier_uniform_(self.q_cov_linear.weight)
        xavier_uniform_(self.out_mean_proj.weight)
        xavier_uniform_(self.out_cov_proj.weight)
        if self.proj_bias:
            constant_(self.v_mean_linear.bias, 0.)
            constant_(self.v_cov_linear.bias, 0.)
            constant_(self.k_mean_linear.bias, 0.)
            constant_(self.k_cov_linear.bias, 0.)
            if hasattr(self, "q_mean_linear"):
                constant_(self.q_mean_linear.bias, 0.)
                constant_(self.q_cov_linear.bias, 0.)
            constant_(self.out_mean_proj.bias, 0.)
            constant_(self.out_cov_proj.bias, 0.)

    def forward(self, query_mean, query_cov, key_mean, key_cov, values_mean, values_cov, mask=None, atten_type='w2', zero_pad=True):
        batch_size = query_mean.size(0)

        # 线性变换和分头 (h 表示多头数)
        value_mean = self.v_mean_linear(values_mean).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value_cov  = self.v_cov_linear(values_cov).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        if self.kq_same:
            key_mean = self.k_mean_linear(key_mean).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            key_cov  = self.k_cov_linear(key_cov).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            query_mean = self.k_mean_linear(query_mean).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            query_cov  = self.k_cov_linear(query_cov).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        else:
            query_mean = self.q_mean_linear(query_mean).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            query_cov  = self.q_cov_linear(query_cov).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            key_mean   = self.k_mean_linear(key_mean).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
            key_cov    = self.k_cov_linear(key_cov).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 注意力计算
        if atten_type == 'w2':
            # 使用分布距离（Wasserstein-2距离，即这里的 NIG 定义距离）计算注意力
            scores_mean, scores_cov = uattention(query_mean, query_cov, key_mean, key_cov,
                                                 value_mean, value_cov, self.d_k, mask, self.dropout, zero_pad, self.gammas)
        else:  # 'dp'
            # 使用点积计算注意力，并考虑不确定性差异
            scores_mean, scores_cov = attention(query_mean, query_cov, key_mean, key_cov,
                                                value_mean, value_cov, self.d_k, mask, self.dropout, zero_pad, self.gammas)

        # 多头拼接并线性变换回输出维度
        concat_mean = scores_mean.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        concat_cov  = scores_cov.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output_mean = self.out_mean_proj(concat_mean)
        output_cov  = self.out_cov_proj(concat_cov)
        return output_mean, output_cov
