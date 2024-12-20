# -*- coding = utf-8 -*-
# @Time : 2024/3/22 10:57
# @Author : cb
# @File :model.py
# @Software : PyCharm
import math
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
from idea.utils import spectral_feature_extraction
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class SineEncoding(nn.Module):
    def __init__(self, hidden_dim=128):
        super(SineEncoding, self).__init__()
        self.constant = 100
        self.hidden_dim = hidden_dim
        self.eig_w = nn.Linear(hidden_dim + 1, hidden_dim)
        self.mha_norm = nn.LayerNorm(128)
        self.ffn_norm = nn.LayerNorm(128)
        self.mha_dropout = nn.Dropout(0.1)
        self.ffn_dropout = nn.Dropout(0.1)
        self.mha = nn.MultiheadAttention(128, 4, 0.1, batch_first=True)
        self.ffn = FeedForwardNetwork(128, 128, 128)
        self.decoder = nn.Linear(128, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, e):
        # input:  [B, N]
        # output: [B, N, d]

        ee = e * self.constant
        div = torch.exp(torch.arange(0, self.hidden_dim, 2) * (-math.log(10000) / self.hidden_dim)).to(e.device)
        pe = ee.unsqueeze(2) * div
        eeig = torch.cat((e.unsqueeze(2), torch.sin(pe), torch.cos(pe)), dim=2)
        eig = self.eig_w(eeig)
        mha_eig = self.mha_norm(eig)
        mha_eig, attn = self.mha(mha_eig, mha_eig, mha_eig)
        eig = eig + self.mha_dropout(mha_eig)

        ffn_eig = self.ffn_norm(eig)
        ffn_eig = self.ffn(ffn_eig)
        eig = eig + self.ffn_dropout(ffn_eig)
        new_e = self.decoder(eig).squeeze()
        # new_e = torch.sigmoid(new_e)

        return new_e


class CrossAttention(nn.Module):
    def __init__(self, in_features):
        super(CrossAttention, self).__init__()

        # self.num_heads = num_heads

        self.query = nn.Linear(90, 90)
        self.key = nn.Linear(90, 90)
        self.value = nn.Linear(90, 90)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        # b, h, w = x1.size()
        # 计算query、key和value
        # x1 = x1.transpose(1, 2)
        # x2 = x2.transpose(1, 2)
        q = self.query(x1)
        k = self.key(x2)
        v = self.value(x2)

        # 计算注意力权重
        attention_A = torch.matmul(q, k.transpose(1, 2)) / torch.sqrt(torch.tensor(90, dtype=torch.float32))
        # attention_A = torch.matmul(k.transpose(1, 2), q)
        attention_A = F.softmax(attention_A, dim=1)

        # 使用注意力权重对value进行加权求和
        f = torch.matmul(attention_A, v)

        return f


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))

    def forward(self, x):
        batch_size = x.shape[0]

        # Split the embedding dimension into multiple heads
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention with multi-heads
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Concatenate the multiple heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1,
                                                                              self.num_heads * self.head_dim)

        # Final linear layer
        output = self.fc_out(attention_output)

        return output


class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.scale = torch.sqrt(torch.FloatTensor([embed_dim]) / 9)

    def forward(self, x):
        device = x.device
        self.scale = self.scale.to(device)

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-2)
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output + x
        return attention_output

class AdaptiveFusionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AdaptiveFusionLayer, self).__init__()
        self.W1 = nn.Linear(input_dim, 1)  # Linear transformation for H1
        self.W2 = nn.Linear(input_dim, 1)  # Linear transformation for H2

    def forward(self, H1, H2):
        # H1 and H2: batch_size x 90 x 90
        batch_size, num_nodes, feature_dim = H1.size()

        # 对 H1 和 H2 进行线性变换
        H1_proj = self.W1(H1)  # batch_size x 90 x 1
        H2_proj = self.W2(H2)  # batch_size x 90 x 1

        # 将 H1_proj 和 H2_proj 拼接在一起
        H_concat = torch.cat([H1_proj, H2_proj], dim=2)  # batch_size x 90 x 2

        # 计算 softmax 注意力权重
        H_softmax = F.softmax(F.leaky_relu(H_concat), dim=2)  # batch_size x 90 x 2

        # 将注意力权重分给 H1 和 H2
        H1_weighted = H1 * H_softmax[:, :, 0].unsqueeze(2)  # batch_size x 90 x 180
        H2_weighted = H2 * H_softmax[:, :, 1].unsqueeze(2)  # batch_size x 90 x 180

        # 计算最终的融合输出
        H_output = H1_weighted + H2_weighted  # batch_size x 90 x 180

        return H_output

class AttentionFilter(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(AttentionFilter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, num_features]
        batch_size, num_nodes, _ = x.shape
        attn_weights = []
        attn_weights = []
        energy = torch.sum(x ** 2, dim=-1)
        energy_weights = torch.sigmoid(energy)
        for i in range(num_nodes):  # 对每个频率行计算权重
            freq_feature = x[:, i, :]  # 取出该行
            hidden = torch.relu(self.fc1(freq_feature))  # 第一层线性变换 + 激活
            score = self.fc2(hidden)
            w = torch.sigmoid(score)  # 使用sigmoid确保权重在0到1之间
            attn_weights.append(w)

        attn_weights = torch.cat(attn_weights, dim=1)  # 将每列的权重拼接成一个向量
        attn_weights = attn_weights * energy_weights
        Filter_matrix = torch.diag_embed(attn_weights)  # [batch_size, num_features, num_features]
        return Filter_matrix, attn_weights


class FeedForwardNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForwardNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 交叉注意力层
        self.cross_attention = CrossAttention(64)
        self.attention = SelfAttention(90)

        self.eig_encoder = SineEncoding(128)


        self.bn = nn.BatchNorm2d(90)

        # Conv2D layers9
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(1, 90))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(90, 1))

        self.fusion = AdaptiveFusionLayer(90)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.AttFilter = AttentionFilter(90, 32)
        # self.AttFilter = AttentionFilter(64, 32)

        # self.fasterNetblock = FasterNetBlock(64)

        # self.SAtt = nn.Sequential(
        #     nn.Linear(90, 90 // 10),
        #     nn.ReLU(),
        #     nn.Linear(90 // 10, 90),
        #     nn.Sigmoid(),
        # )

        self.mlp = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 3),
        )

    def forward(self, dti, fmri, ids):

        h1, l1 = spectral_feature_extraction(dti, 0)
        h2, l2 = spectral_feature_extraction(fmri, 0.5)

        # H = [h1, h2]
        eig1 = self.eig_encoder(l1)
        eig2 = self.eig_encoder(l2)

        f1 = h1.transpose(1, 2) @ dti
        f2 = h2.transpose(1, 2) @ fmri

        f1 = self.attention(f1)
        f2 = self.attention(f2)
        # filter = low_pass_filter_matrix(90, 90)

        filter1, w1 = self.AttFilter(f1)
        filter2, w2 = self.AttFilter(f2)

        # filter0 = [w1 * eig1, w2 * eig2]
        # F = [f1, f2]

        # Filter2 = filter2
        Filter1 = torch.diag_embed(w1 * eig1)
        Filter2 = torch.diag_embed(w2 * eig2)
        #
        x1 = h1 @ Filter1 @ f1
        x2 = h2 @ Filter2 @ f2

        fusion = self.fusion(x1, x2)
        f = fusion @ fusion.transpose(1, 2)
        fs = f.unsqueeze(1)
        fs_output = self.conv1(fs)
        output = self.conv2(fs_output).squeeze()
        output = self.mlp(output)


        return output, torch.exp(x1.mean(dim=2)), torch.exp(x2.mean(dim=2))
        # return output
        # return output
