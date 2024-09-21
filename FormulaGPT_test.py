# ======================================
# === Pytorch手写Transformer完整代码
# ======================================

import math

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from set_encoder import SetEncoder
# from src.nesymres.architectures.beam_search import BeamHypotheses
import numpy as np
import copy
import json
# from numpy import *
acc = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def pad_sequence(sequence, target_length=128, padding_token='P'):
    # 使用空格将序列拆分成单词列表
    words = sequence.split(' ')

    # 计算需要填充的单词数量
    num_padding = target_length - len(words)

    # 如果单词数量已经足够，只返回前target_length个单词
    if num_padding <= 0:
        return ' '.join(words[:target_length])

    # 否则，添加填充并返回
    padded_words = words + [padding_token] * num_padding
    return ' '.join(padded_words)


def read_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading from file: {e}")
        return None


tgt_vocab = {'P': 0, '+': 1, '-': 2, '*': 3, '/': 4, 'sin': 5, 'cos': 6, 'exp': 7, 'var_x1': 8,
             '0.1': 9, '0.2': 10, '0.3': 11, '0.4': 12, '0.5': 13, '0.6': 14, '0.7': 15, '0.8': 16, '0.9': 17,
             '1.0': 18, '0.0': 19, 'S': 20, 'E': 21}

idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# print('idx2word',idx2word)
tgt_vocab_size = len(tgt_vocab)

# transformer epochs
# epochs = 400
epochs = 1000
src_len = 8  # （源句子的长度）enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length
# formulaGPT-1000
# Transformer Parameters
d_model = 512  # Embedding Size（token embedding和position编码的维度）
# FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_ff = 2048
d_k = d_v = 252*1  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 8 # number of Encoder of Decoder Layer（Block的个数）
n_heads = 8 * 2  # number of heads in Multi-Head Attention（有几套头）



# ====================================================================================================
# Transformer模型

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  ##5000*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  ##5000 * 1
        # print("position",position.shape)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        # print('div_term',div_term.shape) #[256] = d_model/2
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)  # shape = [5000，1，512]
        # print(pe.shape)
        self.register_buffer('pe', pe)  ##通过register_buffer()注册过的张量：会自动成为模型中的参数，随着模型移动（gpu/cpu）而移动，但是不会随着梯度进行更新。

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        # print('x',x.shape)
        # print(self.pe[:x.size(0), :].shape)
        x = x + self.pe[:x.size(0), :]
        # print('x',x.shape)
        return self.dropout(x)


def get_attn_pad_mask(seq_q, seq_k):  ##(encode 和 deconde都用)
    # pad mask的作用：在对value向量加权平均的时候，可以让pad对应的alpha_ij=0，这样注意力就不会考虑到pad向量
    """这里的q,k表示的是两个序列（跟注意力机制的q,k没有关系），例如encoder_inputs (x1,x2,..xm)和encoder_inputs (x1,x2..xm)
    encoder和decoder都可能调用这个函数，所以seq_len视情况而定
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # 这个seq_q只是用来expand维度的
    # print('seq_k.size()', seq_k.size())
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  ## seq_k.data.eq(0) seq_k中等于0的为true。
    # print('22',pad_attn_mask)
    # print('33',pad_attn_mask.expand(batch_size, len_q, len_k))
    # [batch_size, len_q, len_k] 构成一个立方体(batch_size个这样的矩阵)
    return pad_attn_mask.expand(batch_size, len_q, len_k)


def get_attn_subsequence_mask(seq):  ## 只在decode用
    """建议打印出来看看是什么的输出（一目了然）
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    # attn_shape: [batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # 生成一个上三角矩阵
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
    # print('44',subsequence_mask)
    return subsequence_mask  # [batch_size, tgt_len, tgt_len]

# ==========================================================================================
class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V, attn_mask):
        """
        Q: [batch_size, n_heads, len_q, d_k]
        K: [batch_size, n_heads, len_k, d_k]
        V: [batch_size, n_heads, len_v(=len_k), d_v]
        attn_mask: [batch_size, n_heads, seq_len, seq_len]
        说明：在encoder-decoder的Attention层中len_q(q1,..qt)和len_k(k1,...km)可能不同
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
                 np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask矩阵填充scores（用-1e9填充scores中与attn_mask中值为1位置相对应的元素）
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # 对最后一个维度(v)做softmax
        # print(attn)
        # scores : [batch_size, n_heads, len_q, len_k] * V: [batch_size, n_heads, len_v(=len_k), d_v]
        # context: [batch_size, n_heads, len_q, d_v]
        context = torch.matmul(attn, V)
        # context：[[z1,z2,...],[...]]向量, attn注意力稀疏矩阵（用于可视化的）
        return context, attn


class MultiHeadAttention(nn.Module):
    """这个Attention类可以实现:
    Encoder的Self-Attention
    Decoder的Masked Self-Attention
    Encoder-Decoder的Attention
    输入：seq_len x d_model
    输出：seq_len x d_model
    """

    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads,
                             bias=False)  # q,k必须维度相同，不然无法做点积
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # 下面的多头的参数矩阵是放在一起做线性变换的，然后再拆成多个头，这是工程实现的技巧
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)
        #           线性变换               拆成多头

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # K和V的长度一定相同，维度可以不同
        K = self.W_K(input_K).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   n_heads, d_v).transpose(1, 2)

        # 因为是多头，所以mask矩阵要扩充成4维的
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # 下面将不同头的输出向量拼接在一起
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, n_heads * d_v)

        # 这个全连接层可以保证多头attention的输出仍然是seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn


# Pytorch中的Linear只会对最后一维操作，所以正好是我们希望的每个位置用同一个全连接网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):
        """
        inputs: [batch_size, seq_len, d_model]
        """
        residual = inputs
        output = self.fc(inputs)
        # [batch_size, seq_len, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual)


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        """E
        enc_inputs: [batch_size, src_len, d_model]
        enc_self_attn_mask: [batch_size, src_len, src_len]  mask矩阵(pad mask or sequence mask)
        """
        # enc_outputs: [batch_size, src_len, d_model], attn: [batch_size, n_heads, src_len, src_len]
        # 第一个enc_inputs * W_Q = Q
        # 第二个enc_inputs * W_K = K
        # 第三个enc_inputs * W_V = V
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V（未线性变换前）
        enc_outputs = self.pos_ffn(enc_outputs)
        # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn


class DecoderLayer(nn.Module):
    def __init__(self):
        super(DecoderLayer, self).__init__()
        self.dec_self_attn = MultiHeadAttention()
        self.dec_enc_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, dec_inputs, enc_outputs, dec_self_attn_mask, dec_enc_attn_mask):
        """
        dec_inputs: [batch_size, tgt_len, d_model]
        enc_outputs: [batch_size, src_len, d_model]
        dec_self_attn_mask: [batch_size, tgt_len, tgt_len]
        dec_enc_attn_mask: [batch_size, tgt_len, src_len]
        """
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len]
        dec_outputs, dec_self_attn = self.dec_self_attn(dec_inputs, dec_inputs, dec_inputs,
                                                        dec_self_attn_mask)  # 这里的Q,K,V全是Decoder自己的输入
        # print('dec_self_attn_mask',dec_self_attn_mask)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  # Attention层的Q(来自decoder) 和 K,V(来自encoder)
        # print('dec_enc_attn_mask',dec_enc_attn_mask)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        # dec_self_attn, dec_enc_attn这两个是为了可视化的
        return dec_outputs, dec_self_attn, dec_enc_attn


class Config:
    def __init__(self, N_p, activation, bit16, dec_layers, dec_pf_dim, dim_hidden,
                 dim_input, dropout, input_normalization, length_eq,
                 linear,
                 ln,
                 lr,
                 mean,
                 n_l_enc,
                 norm,
                 num_features,
                 num_heads,
                 num_inds,
                 output_dim,
                 sinuisodal_embeddings,
                 src_pad_idx,
                 std,
                 trg_pad_idx):
        self.N_p = N_p
        self.activation = activation
        self.bit16 = bit16
        self.dec_layers = dec_layers
        self.dec_pf_dim = dec_pf_dim
        self.dim_hidden = dim_hidden
        self.dim_input = dim_input  ## 输入维度
        self.dropout = dropout
        self.input_normalization = input_normalization
        self.length_eq = length_eq
        self.linear = linear
        self.ln = ln
        self.lr = lr
        self.mean = mean
        self.n_l_enc = n_l_enc
        self.norm = norm
        self.num_features = num_features  #### 输出行数 num_features * 512
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.output_dim = output_dim
        self.sinuisodal_embeddings = src_pad_idx
        self.src_pad_idx = src_pad_idx
        self.std = std
        self.trg_pad_idx = trg_pad_idx


# 创建配置对象并设置属性
cfg = Config(0, 'relu', True, 5, 512, 512, 4, 0, False, 60, False, True, 0.0001, 0.5, 5, True, 4, 8, 50, 60,
             False, 0, 0.5, 0)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding 总共几种字符，就生成几（src_vocab_size）种编码
        self.src_emb = SetEncoder(cfg)
        self.pos_emb = PositionalEncoding(
            d_model)  # Transformer中位置编码是固定的，不需要学习
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """

        enc_outputs = self.src_emb(
            enc_inputs)  # [batch_size, src_len, d_model] ;enc_input作为输入，会在Embeding中进行一些列运算映射的。

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1)  # [batch_size, src_len, d_model] ### 得到加上位置编码的输入序列
        # Encoder输入序列的pad mask矩阵
        # enc_self_attn_mask = get_attn_pad_mask(
        #     enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attn_mask = torch.zeros(enc_outputs.size()[0], enc_outputs.size()[1], 4, dtype=torch.bool).cuda()
        # print('enc_self_attn_mask',enc_self_attn_mask)
        enc_self_attns = []  # 在计算中不需要用到，它主要用来保存你接下来返回的attention的值（这个主要是为了你画热力图等，用来看各个词之间的关系
        for layer in self.layers:  # for循环访问nn.ModuleList对象
            # 上一个block的输出enc_outputs作为当前block的输入
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # 传入的enc_outputs其实是input，传入mask矩阵是因为你要做self attention
            enc_self_attns.append(enc_self_attn)  # 这个只是为了可视化
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(
            tgt_vocab_size, d_model)  # Decoder输入的embed词表
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer()
                                     for _ in range(n_layers)])  # Decoder的blocks

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(
            dec_inputs)  # [batch_size, tgt_len, d_model]
        # print('dec_outputs',dec_outputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        # Decoder输入序列的pad mask矩阵（这个例子中decoder是没有加pad的，实际应用中都是有pad填充的）
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：当前时刻是看不到未来的信息的
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]

        # Decoder中把两种mask矩阵相加（既屏蔽了pad的信息，也屏蔽了未来时刻的信息）
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(device)  # [batch_size, tgt_len, tgt_len]; torch.gt比较两个矩阵的元素，大于则返回1，否则返回0
        # 这个mask主要用于encoder-decoder attention层
        # get_attn_pad_mask主要是enc_inputs的pad mask矩阵(因为enc是处理K,V的，求Attention时是用v1,v2,..vm去加权的，要把pad对应的v_i的相关系数设为0，这样注意力就不会关注pad向量)
        #                       dec_inputs只是提供expand的size的
        enc_inputs_mask = torch.ones(enc_inputs.size()[0], enc_inputs.size()[2]).cuda()
        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs_mask)  # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # Decoder的Block是上一个Block的输出dec_outputs（变化）和Encoder网络的输出enc_outputs（固定）
            dec_outputs, dec_self_attn, dec_enc_attn = layer(dec_outputs, enc_outputs, dec_self_attn_mask,
                                                             dec_enc_attn_mask)
            dec_self_attns.append(dec_self_attn)
            dec_enc_attns.append(dec_enc_attn)
        # dec_outputs: [batch_size, tgt_len, d_model]
        return dec_outputs, dec_self_attns, dec_enc_attns

class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.projection = nn.Linear(
            d_model, tgt_vocab_size, bias=False).to(device)

    def forward(self, enc_inputs, dec_inputs):
        """Transformers的输入：两个序列
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # 经过Encoder网络后，得到的输出还是[batch_size, src_len, d_model]
        enc_outputs, enc_self_attns = self.encoder(enc_inputs)
        # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attns: [n_layers, batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [n_layers, batch_size, tgt_len, src_len]
        dec_outputs, dec_self_attns, dec_enc_attns = self.decoder(
            dec_inputs, enc_inputs, enc_outputs)
        # dec_outputs: [batch_size, tgt_len, d_model] -> dec_logits: [batch_size, tgt_len, tgt_vocab_size]
        dec_logits = self.projection(dec_outputs)
        return dec_logits.view(-1, dec_logits.size(-1)), enc_self_attns, dec_self_attns, dec_enc_attns

def Arity(s):
    if s in ['x', 'var_x1', 'var_x2', 'var_x3', 'var_x4', 'var_x5', 'var_x6', 'var_x7', 'var_x8', 'var_x9', 'c']:
        return 0
    if s in ['sin', 'cos', 'exp', 'ln', 'sqrt']:
        return 1
    if s in ['+', '-', '*', '/', '^']:
        return 2


def R2(y, y_pred):
    return 1 - np.mean((y - y_pred) ** 2) / np.mean((y - np.mean(y)) ** 2)


def process_R2(R2, threshold=1):
    if R2 <= 0 or threshold > 1:
        return str(float(0))
    else:
        return str(float(round(R2, 1)))


def all_farward(l2, X):
    global stack1, v1
    stack1 = []
    for i in range(len(l2)):
        s = l2[-(i + 1)]
        if s == 'var_x1':
            stack1.append(X)
        if s == '0':
            mkl = 0
        if s in ['sin', 'cos', 'ln', 'exp', 'sqrt']:
            if s == 'exp':
                v1 = np.exp(stack1.pop())
            if s == 'log':
                v1 = np.log(stack1.pop())
            if s == 'cos':
                v1 = np.cos(stack1.pop())
            if s == 'sin':
                v1 = np.sin(stack1.pop())
            if s == 'sqrt':
                v1 = np.sqrt(stack1.pop())
            stack1.append(v1)
        if s in ['+', '-', '*', '/']:
            if s == '+':
                v1 = stack1.pop() + stack1.pop()
            if s == '-':
                v1 = stack1.pop() - stack1.pop()
            if s == '*':
                v1 = stack1.pop() * stack1.pop()
            if s == '/':
                v1 = stack1.pop() / stack1.pop()
            stack1.append(v1)
    return stack1[0]


model = Transformer().to(device)
# 这里的损失函数里面设置了一个参数 ignore_index=0，因为 "pad" 这个单词的索引为 0，这样设置以后，就不会计算 "pad" 的损失（因为本来 "pad" 也没有意义，不需要计算）
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-5,
                      momentum=0.99)  # 用adam的话效果不好
# optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-5)
# ====================================================================================================
LOSS = []

# 指定之前保存权重的位置
model_path = 'formulaGPT-epoch-10000.pth'  # 更新为你的文件路径
# model_path = 'formulaGPT-epoch-10000-2.pth'  # 更新为你的文件路径

# 加载权重到模型中
model.load_state_dict(torch.load(model_path))
# 设置为评估模式
model.eval()

def beam_search_decoder(model, enc_input, beam_width, max_length, sos_token, eos_token):
    """
    Beam search decoder.
    model: 一个函数，用于给定输入生成输出分布。
    input: 输入数据。
    beam_width: 束宽度。
    max_length: 输出的最大长度。
    sos_token: 开始符号的id。
    eos_token: 结束符号的id。
    返回: 最佳序列。
    """
    # 初始化束: 每个元素是(序列, 分数)
    beams = [(torch.tensor([sos_token]), 0.0, 0.0)]

    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    for _ in range(max_length):
        candidates = []
        for ssr in beams:
            # print('ssr',ssr)
            seq, score, r2_score = ssr
            Ari = np.ones(beam_width)
            # print('seq',seq)
            Ari_n = 1
            next_score = torch.tensor(0.0)
            fum = []
            for i in range(len(seq.tolist())): ##读取序列中新的表达式部分
                ss = idx2word[seq.tolist()[-(i+1)]]
                # print('9'*10,ss)
                if ss in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
                    fum.append(ss)
                if ss in ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
                    break
            fum = fum[::-1] ##倒序一下
            # print('1'*20,fum,Ari_n)
            if fum != []:
                for i in range(len(fum)): ### 计算现有的符号的Ari
                    Ari_n = Ari_n + Arity(fum[i]) - 1
            Ari = Ari*Ari_n ## 得到现有序列的 Ari
            # print('8'*10,fum,Ari)
            if seq[-1] == eos_token:
                # 如果序列已经结束，直接加入候选
                candidates.append((seq, score, r2_score))
                continue

            # 使用模型预测下一个词的分布
            # probs = model(seq, input)
            # print('4444', seq.unsqueeze(0))
            dec_outputs, _, _ = model.decoder(seq.unsqueeze(0).to(device), enc_input, enc_outputs)
            # print('4444', dec_outputs)
            projected = model.projection(dec_outputs)
            probs = projected.squeeze(0)[-1]
            # print('4444', probs)
            # probs = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
            probs = torch.softmax(probs,dim=0)  ## 得到选择每个符号的概率
            if Ari.any() != 0:
                probs[9:20] = -torch.inf
                probs[0] = -torch.inf
            # 取概率最高的beam_width个词

            # print('44445555',probs)
            top_probs, top_indices = torch.topk(probs, beam_width) ##选择概率最大的前k个。
            # print('top_probs', top_probs)
            sx = enc_input[:, :, 0].cpu().numpy()[0]
            sy = enc_input[:, :, -1].cpu().numpy()[0]  ##### 此处要改 y的位置应该用-1为妥
            # print('next_seq.tolist', next_seq.tolist())
            # 为每个词创建新的序列并更新其得分
            for i in range(beam_width):
                if Ari[i] != 0:
                    sym = idx2word[top_indices[i].item()]
                    # print(sym,str(i))
                    if sym in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
                        # print('5'*10,sym)
                        fum.append(sym)
                        # Ari[i] = Ari[i] + Arity(sym) - 1
                        # print('3' * 10, Ari[i], str(i))
                    next_seq = torch.cat([seq.to(device), top_indices[i].unsqueeze(0).to(device)],-1)
                    # print('top_indices[i].unsqueeze(0)',top_indices[i].unsqueeze(0))
                if Ari[i] == 0:
                    # print('5'*10,Ari[i],fum,str(i))
                    # result_seq = [idx2word[i] for i in next_seq.tolist()]

                    y_pre = all_farward(fum, sx)
                    # print('y&y_pre',sy, y_pre)
                    r2 = R2(sy, y_pre)
                    s_r2 = process_R2(r2)
                    # print('s_r2', s_r2)
                    next_word = tgt_vocab[s_r2]
                    # print(fum,s_r2)
                    # print('next_word',next_word)
                    next_seq = torch.cat([seq.to(device), torch.tensor([next_word]).to(device)], -1)
                    if s_r2 == '1.0':
                        # print("over"*10)
                        next_seq = torch.cat([next_seq.to(device), torch.tensor([21]).to(device)], -1)
                    # Ari[i] = 1

                    r2_score = torch.max(torch.tensor(r2_score).clone().detach(),torch.tensor(r2).clone().detach())
                    # r2_score = torch.max(r2_score.clone().detach(), r2.clone().detach())

                #     next_score = score
                #     print('next_score'*5,next_score)
                next_score = score + torch.log(top_probs[i])
                # print('2'*20,r2_score)
                candidates.append((next_seq, next_score, r2_score))
                # print('candidates',candidates)
        # 保留总得分最高的beam_width个候选序列
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # 从束中选择得分最高的序列作为输出
    return max(beams, key=lambda x: x[2])[0]

def greedy_decoder(model, enc_input, start_symbol):
    """贪心编码
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.
    Starting Reference: http://nlp.seas.harvard.edu/2018/04/03/attention.html#greedy-decoding
    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    # print('enc_inputs.size()3333', enc_input.size())
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # 初始化一个空的tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0, dtype=torch.int64)

    terminal = False
    next_symbol = start_symbol
    Ari = 1
    fum = []
    global acc  # 声明全局变量
    max_length = 30 ## 公式最大长度
    nu_c = 0 ## 最多寻找几个最大序列长度
    best_r2 = -np.inf
    best_expression = ['sin','var_x1']
    while not terminal:
        # print('dec_input', dec_input)
        # 预测阶段：dec_input序列会一点点变长（每次添加一个新预测出来的单词）
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=torch.int64).to(device)],
                              -1)
        # print('enc_inputs.size()8888',enc_inputs.size())
        # enc_inputs_mask2 = torch.ones(enc_inputs.size()[0], enc_inputs.size()[2]).cuda()

        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        # print('5' * 100)
        projected = model.projection(dec_outputs)
        # print('projected1111',projected.size())
        if Ari != 0:
            projected[:, -1, 9:20] = -torch.inf
            projected[:, -1, 0] = -torch.inf

        if Ari + len(fum) >= max_length - 2:
            # print('2'*100)
            projected[:, -1, 8] = torch.inf
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        # print('projected[-1]', projected[:,-1,9:20])
        # print('prob',prob)
        # 增量更新（我们希望重复单词预测结果是一样的）
        # 我们在预测时会选择性忽略重复的预测的词，只摘取最新预测的单词拼接到输入序列中
        # 拿出当前预测的单词(数字)。我们用x'_t对应的输出z_t去预测下一个单词的概率，不用z_1,z_2..z_{t-1}
        next_word = prob.data[-1]

        # print('next_word',next_word.item())
        sym = idx2word[next_word.item()]
        if sym in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
            fum.append(sym)
            Ari = Ari + Arity(sym) - 1

        sx = enc_input[:, :, 0].cpu().numpy()[0]
        sy = enc_input[:, :, -1].cpu().numpy()[0]  ##### 此处要改 y的位置应该用-1为妥
        if Ari == 0:
            y_pre = all_farward(fum, sx)
            r2 = R2(sy, y_pre)

            print('r2',r2)
            next_symbol = next_word
            dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=torch.int64).to(device)],
                                  -1)
            s_r2 = process_R2(r2)
            print('s_r2', s_r2)
            next_word = tgt_vocab[s_r2]
            # print('y_pre',y_pre)
            print('Expresssion', fum)
            if r2 >= best_r2:
                best_r2 = r2
                best_expression = fum

                # print('best_r2',best_r2)
                # print('best_expression',best_expression)
            if next_word == tgt_vocab["1.0"]:  ### 如果得到的表达式r2为 1 了就停止搜索
                # print('1'*100)
                acc = acc + 1
                next_symbol = next_word
                dec_input = torch.cat(
                    [dec_input.to(device), torch.tensor([[next_symbol]], dtype=torch.int64).to(device)],
                    -1)
                next_word = tgt_vocab["E"]
            fum = []
            Ari = 1
        # print('next_word',next_word)
        # print('3' * 10, len(dec_input[0]))
        if len(dec_input[0])<=128:
            next_symbol = next_word
        else:
            nu_c = nu_c + 1
            dec_input = torch.zeros(1, 0, dtype=torch.int64)
            next_symbol = tgt_vocab["S"]
        # if next_symbol == tgt_vocab["E"] and s_r2 == '1.0':
        if next_symbol == tgt_vocab["E"] or nu_c >= 20:
            terminal = True
        # print(next_word)

    greedy_dec_predict = dec_input[:, 1:]  # 去除开始符号
    return greedy_dec_predict, best_r2, best_expression

# ==========================================================================================
# 预测阶段
def merge_tensors(D, A, B):
    # 确保A和B是二维的
    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    # 确保D至少有足够的空间来存储A和B
    d_rows, d_cols = D.shape
    if d_rows < a_rows or d_cols < a_cols + b_cols:
        raise ValueError("Tensor D is not large enough to hold the merged tensors A and B.")
    # 将A赋值到D的左侧
    D[:a_rows, :a_cols] = A
    # 将B赋值到D的右侧
    D[:b_rows, -b_cols:] = B
    return D

sample = torch.zeros(50, 4)
N_sample_point = 50
#### 指定测试数据 ####
X = np.linspace(-4, 4, num=N_sample_point)
X= X.reshape(N_sample_point,1)
# X = abs(X)
x1 = X[:,0]
y = np.sin(x1**2) + x1

enc_inputs = merge_tensors(sample, torch.tensor(x1), torch.tensor(y)).unsqueeze(0).to(device)

print("=" * 50)
print("通过训练好的formulaGPT预测的数学公式：")

for i in range(len(enc_inputs)):
    for kk in range(1):
        greedy_dec_predict, Best_r2, Best_expression = greedy_decoder(model, enc_inputs[i].unsqueeze(0).to(device), start_symbol=tgt_vocab["S"])
        # greedy_dec_predict = beam_search_decoder(model, enc_inputs[i].unsqueeze(0).to(device), 4, 90, tgt_vocab["S"], tgt_vocab["E"])
        print('################### The Best Expression ###################')
        print('The best $R^2$:', Best_r2)
        print('The best expression: ', Best_expression)


