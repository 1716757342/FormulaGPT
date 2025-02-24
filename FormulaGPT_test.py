# ======================================
# === FormulaGPT inference code
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
import numpy as np
import copy
import json


model_pth = 'formulaGPT-epoch-10000.pth' #### Specifies the model path to load
acc = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tgt_vocab = {'P': 0, '+': 1, '-': 2, '*': 3, '/': 4, 'sin': 5, 'cos': 6, 'exp': 7, 'var_x1': 8,
             '0.1': 9, '0.2': 10, '0.3': 11, '0.4': 12, '0.5': 13, '0.6': 14, '0.7': 15, '0.8': 16, '0.9': 17,
             '1.0': 18, '0.0': 19, 'S': 20, 'E': 21}

idx2word = {i: w for i, w in enumerate(tgt_vocab)}
tgt_vocab_size = len(tgt_vocab)

src_len = 8  # enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length
# formulaGPT-1000
# Transformer Parameters
d_model = 512  # Embedding Size
# FeedForward dimension
d_ff = 2048
d_k = d_v = 252*1  # dimension of K(=Q), V
n_layers = 8 # number of Encoder of Decoder Layer
n_heads = 8 * 2  # number of heads in Multi-Head Attention
# ====================================================================================================
# Transformer model
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
        self.register_buffer('pe', pe)  ## Tensors registered with register_buffer() : automatically become parameters in the model, move as the model moves (gpu/cpu), but are not updated with gradients.

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    # What pad mask does: We can set alpha_ij=0 for pad when averaging the value vectors so that our attention doesn't take into account the pad vectors
    """ Here q,k refers to two sequences (unrelated to attention q,k), such as encoder_inputs (x1,x2,.. xm) and encoder_inputs (x1,x2.. xm)
    Both encoder and decoder may call this function, so seq_len depends
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """
    batch_size, len_q = seq_q.size()  # This seq_q is only used to expand the dimension
    # print('seq_k.size()', seq_k.size())
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # 例如:seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  ## Seq_k.data.eq (0) Anything equal to 0 in seq_k is true.
    # [batch_size, len_q, len_k] Seq_k.data.eq (0) Anything equal to 0 in seq_k is true.
    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq):  ## Only for decoding
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Generate an upper triangular matrix
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
        Note: In the encoder-decoder Attention layer len_q(q1,.. qt) and len_k(k1,... km) may be different
        """
        scores = torch.matmul(Q, K.transpose(-1, -2)) / \
                 np.sqrt(d_k)  # scores : [batch_size, n_heads, len_q, len_k]
        # mask matrix padding scores (padding scores with -1e9 corresponding to 1 in attn_mask)
        # Fills elements of self tensor with value where mask is True.
        scores.masked_fill_(attn_mask, -1e9)

        attn = nn.Softmax(dim=-1)(scores)  # Do softmax on the last dimension (v)
        context = torch.matmul(attn, V)
        return context, attn

class MultiHeadAttention(nn.Module):
    """ This Attention class can be implemented:
    Self-Attention in the Encoder
    Masked Self-Attention for Decoder
    Encoder-Decoder Attention
    Input: seq_len x d_model
    Output: seq_len x d_model
    """
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads,
                             bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        # This fully connected layer ensures that the output of the multi-head attention is still seq_len * d_model
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        """
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        """
        residual, batch_size = input_Q, input_Q.size(0)
        # The following multi-head parameter matrices are linearly transformed together and then split into multiple heads. This is a technique for engineering implementation
        # B: batch_size, S:seq_len, D: dim
        # (B, S, D) -proj-> (B, S, D_new) -split-> (B, S, Head, W) -trans-> (B, Head, S, W)

        # Q: [batch_size, n_heads, len_q, d_k]
        Q = self.W_Q(input_Q).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # K: [batch_size, n_heads, len_k, d_k] # The length of K and V must be the same, and the dimensions can be different
        K = self.W_K(input_K).view(batch_size, -1,
                                   n_heads, d_k).transpose(1, 2)
        # V: [batch_size, n_heads, len_v(=len_k), d_v]
        V = self.W_V(input_V).view(batch_size, -1,
                                   n_heads, d_v).transpose(1, 2)

        # Because of the multiple heads, the mask matrix needs to be expanded to four dimensions
        # attn_mask: [batch_size, seq_len, seq_len] -> [batch_size, n_heads, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)

        # context: [batch_size, n_heads, len_q, d_v], attn: [batch_size, n_heads, len_q, len_k]
        context, attn = ScaledDotProductAttention()(Q, K, V, attn_mask)
        # The output vectors of the different heads are concatenated together below
        # context: [batch_size, n_heads, len_q, d_v] -> [batch_size, len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(
            batch_size, -1, n_heads * d_v)
        # This fully connected layer ensures that the output of the multi-head attention is still seq_len x d_model
        output = self.fc(context)  # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).to(device)(output + residual), attn

# Linear in Pytorch only operates on the last dimension, so we want the same fully connected network for every location
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

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
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
                                                        dec_self_attn_mask)  # Here Q,K,V are all the input of the Decoder itself
        # dec_outputs: [batch_size, tgt_len, d_model], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
        dec_outputs, dec_enc_attn = self.dec_enc_attn(dec_outputs, enc_outputs, enc_outputs,
                                                      dec_enc_attn_mask)  # Attention layer Q(from decoder) and K,V(from encoder)
        # [batch_size, tgt_len, d_model]
        dec_outputs = self.pos_ffn(dec_outputs)
        # dec_self_attn and dec_enc_attn are for visualization
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
        self.dim_input = dim_input  ## Input dimension
        self.dropout = dropout
        self.input_normalization = input_normalization
        self.length_eq = length_eq
        self.linear = linear
        self.ln = ln
        self.lr = lr
        self.mean = mean
        self.n_l_enc = n_l_enc
        self.norm = norm
        self.num_features = num_features  #### The number of lines printed is num_features * 512
        self.num_heads = num_heads
        self.num_inds = num_inds
        self.output_dim = output_dim
        self.sinuisodal_embeddings = src_pad_idx
        self.src_pad_idx = src_pad_idx
        self.std = std
        self.trg_pad_idx = trg_pad_idx


# Create a configuration object and set its properties
cfg = Config(0, 'relu', True, 5, 512, 512, 4, 0, False, 60, False, True, 0.0001, 0.5, 5, True, 4, 8, 50, 60,
             False, 0, 0.5, 0)

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # Generate (src_vocab_size) encodings for a total of several characters
        self.src_emb = SetEncoder(cfg)
        self.pos_emb = PositionalEncoding(
            d_model)  # Position encoding is fixed in Transformer and does not need to be learned
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])

    def forward(self, enc_inputs):
        """
        enc_inputs: [batch_size, src_len]
        """

        enc_outputs = self.src_emb(
            enc_inputs)  # [batch_size, src_len, d_model] ;enc_input as input, will be in Embeding for some column operations mapping.

        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(
            0, 1)  # [batch_size, src_len, d_model] ### Get the input sequence with positional encoding
        # The pad mask matrix of the Encoder input sequence
        # enc_self_attn_mask = get_attn_pad_mask(
        #     enc_inputs, enc_inputs)  # [batch_size, src_len, src_len]
        enc_self_attn_mask = torch.zeros(enc_outputs.size()[0], enc_outputs.size()[1], 4, dtype=torch.bool).cuda()
        # print('enc_self_attn_mask',enc_self_attn_mask)
        enc_self_attns = []  # It is not needed in the calculation, it is mainly used to hold the attention value that you return later (this is mainly for you to draw heat maps, etc., to see the relationship between words
        for layer in self.layers:  # The for loop accesses the nn.ModuleList object
            # The output of the previous block enc_outputs is used as the input of the current block
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)  # The enc_outputs is actually input, and the mask matrix is passed because you want to do self attention
            enc_self_attns.append(enc_self_attn)  # This is just for visualization
        return enc_outputs, enc_self_attns


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.tgt_emb = nn.Embedding(
            tgt_vocab_size, d_model)  # The embed vocabulary for the Decoder input
        self.pos_emb = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([DecoderLayer()
                                     for _ in range(n_layers)])

    def forward(self, dec_inputs, enc_inputs, enc_outputs):
        """
        dec_inputs: [batch_size, tgt_len]
        enc_inputs: [batch_size, src_len]
        enc_outputs: [batch_size, src_len, d_model]
        """
        dec_outputs = self.tgt_emb(
            dec_inputs)  # [batch_size, tgt_len, d_model]
        # print('dec_outputs',dec_outputs)
        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        # Decoder input sequence pad mask matrix
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]
        # Masked Self_Attention：There is no future information in the present moment
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]

        # In the Decoder, the two mask matrices are added together (both the information of the pad and the information of the future time are masked).
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(device)  # [batch_size, tgt_len, tgt_len]; torch.gt The elements of the two matrices are compared, and 1 is returned if the value is greater than, and 0 otherwise
        # This mask is mainly used in the encoder-decoder attention layer
        # get_attn_pad_mask is mainly the pad mask matrix of enc_inputs (since enc deals with K,V, Attention is obtained with v1,v2,.. vm is deweighted, and the correlation coefficient of v_i corresponding to pad is set to 0, so that attention is not paid to the pad vector.)

        enc_inputs_mask = torch.ones(enc_inputs.size()[0], enc_inputs.size()[2]).cuda()
        dec_enc_attn_mask = get_attn_pad_mask(
            dec_inputs, enc_inputs_mask)  # [batc_size, tgt_len, src_len]
        dec_self_attns, dec_enc_attns = [], []
        for layer in self.layers:
            # dec_outputs: [batch_size, tgt_len, d_model], dec_self_attn: [batch_size, n_heads, tgt_len, tgt_len], dec_enc_attn: [batch_size, h_heads, tgt_len, src_len]
            # The blocks of the Decoder are the outputs of the previous Block dec_outputs (varying) and the outputs of the Encoder network enc_outputs (fixed).
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
        """ Input to Transformers: Two sequences
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        """
        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size).to(self.device)

        # enc_outputs: [batch_size, src_len, d_model], enc_self_attns: [n_layers, batch_size, n_heads, src_len, src_len]
        # After passing through the Encoder network, the resulting output is again [batch_size, src_len, d_model].
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
# The ignore_index=0 parameter is set in the loss function, because the index of the word "pad" is 0, so the loss for "pad" will not be calculated (because "pad" has no meaning, so we don't need to calculate it).criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.SGD(model.parameters(), lr=1e-5,
                      momentum=0.99)
# ====================================================================================================
LOSS = []

# Specifies where the weights were previously saved
model_path = model_pth # Update to your file path

# load weights into the model
model.load_state_dict(torch.load(model_path))
# Set to evaluation mode
model.eval()

def beam_search_decoder(model, enc_input, beam_width, max_length, sos_token, eos_token):
    """
    Beam search decoder.
    model: A function that generates an output distribution given an input.
    input: Enter the data.
    beam_width: Beam ded width.
    max_length: Maximum length of the output.
    sos_token: The id of the start symbol.
    eos_token: The id of the closing symbol.
    Return: The best sequence.
    """
    # Initialize bundle: each element is (sequence, score)
    beams = [(torch.tensor([sos_token]), 0.0, 0.0)]

    enc_outputs, enc_self_attns = model.encoder(enc_input)

    dec_input = torch.zeros(1, 0).type_as(enc_input.data)
    for _ in range(max_length):
        candidates = []
        for ssr in beams:
            # print('ssr',ssr)
            seq, score, r2_score = ssr
            Ari = np.ones(beam_width)
            Ari_n = 1
            next_score = torch.tensor(0.0)
            fum = []
            for i in range(len(seq.tolist())): ## Read the new expression part of the sequence
                ss = idx2word[seq.tolist()[-(i+1)]]
                # print('9'*10,ss)
                if ss in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
                    fum.append(ss)
                if ss in ['0.0','0.1','0.2','0.3','0.4','0.5','0.6','0.7','0.8','0.9','1.0']:
                    break
            fum = fum[::-1] ## Reverse the order
            # print('1'*20,fum,Ari_n)
            if fum != []:
                for i in range(len(fum)): ### Calculate the existing symbols
                    Ari_n = Ari_n + Arity(fum[i]) - 1
            Ari = Ari*Ari_n ## The Ari of the existing sequence is obtained
            if seq[-1] == eos_token:
                # If the sequence has ended, directly join the candidate
                candidates.append((seq, score, r2_score))
                continue

            # Use the model to predict the distribution of the next word

            dec_outputs, _, _ = model.decoder(seq.unsqueeze(0).to(device), enc_input, enc_outputs)
            projected = model.projection(dec_outputs)
            probs = projected.squeeze(0)[-1]

            probs = torch.softmax(probs,dim=0)  ## The probability of selecting each symbol is obtained
            if Ari.any() != 0:
                probs[9:20] = -torch.inf
                probs[0] = -torch.inf
            # Take the beam_width words with the highest probability

            top_probs, top_indices = torch.topk(probs, beam_width) ##Select the top k with the largest probability。
            # print('top_probs', top_probs)
            sx = enc_input[:, :, 0].cpu().numpy()[0]
            sy = enc_input[:, :, -1].cpu().numpy()[0]

            # Create a new sequence for each word and update its score
            for i in range(beam_width):
                if Ari[i] != 0:
                    sym = idx2word[top_indices[i].item()]
                    if sym in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
                        # print('5'*10,sym)
                        fum.append(sym)
                    next_seq = torch.cat([seq.to(device), top_indices[i].unsqueeze(0).to(device)],-1)
                if Ari[i] == 0:

                    y_pre = all_farward(fum, sx)
                    r2 = R2(sy, y_pre)
                    s_r2 = process_R2(r2)
                    next_word = tgt_vocab[s_r2]
                    next_seq = torch.cat([seq.to(device), torch.tensor([next_word]).to(device)], -1)
                    if s_r2 == '1.0':
                        next_seq = torch.cat([next_seq.to(device), torch.tensor([21]).to(device)], -1)
                    r2_score = torch.max(torch.tensor(r2_score).clone().detach(),torch.tensor(r2).clone().detach())

                next_score = score + torch.log(top_probs[i])
                candidates.append((next_seq, next_score, r2_score))
        # The beam_width candidate sequences with the highest total score are kept
        beams = sorted(candidates, key=lambda x: x[1], reverse=True)[:beam_width]

    # The sequence with the highest score from the bundle is selected as the output
    return max(beams, key=lambda x: x[2])[0]

def greedy_decoder(model, enc_input, start_symbol):
    """Greedy coding
    For simplicity, a Greedy Decoder is Beam search when K=1. This is necessary for inference as we don't know the
    target sequence input. Therefore we try to generate the target input word by word, then feed it into the transformer.

    :param model: Transformer Model
    :param enc_input: The encoder input
    :param start_symbol: The start symbol. In this example it is 'S' which corresponds to index 4
    :return: The target input
    """
    enc_outputs, enc_self_attns = model.encoder(enc_input)
    # Initialize a full tensor: tensor([], size=(1, 0), dtype=torch.int64)
    dec_input = torch.zeros(1, 0, dtype=torch.int64)

    terminal = False
    next_symbol = start_symbol
    Ari = 1
    fum = []
    global acc  # Declaring global variables
    max_length = 30 ## Maximum length of formula
    nu_c = 0 ## Look for at most a few maximum sequence lengths
    best_r2 = -np.inf
    best_expression = ['sin','var_x1']
    while not terminal:
        # Prediction phase: The dec_input sequence gets longer (adding a new predicted word at a timePrediction phase: The dec_input sequence gets longer (adding a new predicted word at a time)
        dec_input = torch.cat([dec_input.to(device), torch.tensor([[next_symbol]], dtype=torch.int64).to(device)],
                              -1)

        dec_outputs, _, _ = model.decoder(dec_input, enc_input, enc_outputs)
        projected = model.projection(dec_outputs)
        if Ari != 0:
            projected[:, -1, 9:20] = -torch.inf
            projected[:, -1, 0] = -torch.inf

        if Ari + len(fum) >= max_length - 2:
            projected[:, -1, 8] = torch.inf
        prob = projected.squeeze(0).max(dim=-1, keepdim=False)[1]
        next_word = prob.data[-1]
        sym = idx2word[next_word.item()]
        if sym in ['+', '-', '*', '/', 'sin', 'cos', 'exp', 'var_x1']:
            fum.append(sym)
            Ari = Ari + Arity(sym) - 1

        sx = enc_input[:, :, 0].cpu().numpy()[0]
        sy = enc_input[:, :, -1].cpu().numpy()[0]
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
            if next_word == tgt_vocab["1.0"]:  ### If the expression r2 is 1, stop searching
                acc = acc + 1
                next_symbol = next_word
                dec_input = torch.cat(
                    [dec_input.to(device), torch.tensor([[next_symbol]], dtype=torch.int64).to(device)],
                    -1)
                next_word = tgt_vocab["E"]
            fum = []
            Ari = 1

        if len(dec_input[0])<=128:
            next_symbol = next_word
        else:
            nu_c = nu_c + 1
            dec_input = torch.zeros(1, 0, dtype=torch.int64)
            next_symbol = tgt_vocab["S"]
        if next_symbol == tgt_vocab["E"] or nu_c >= 20:
            terminal = True

    greedy_dec_predict = dec_input[:, 1:]  # Remove start symbol
    return greedy_dec_predict, best_r2, best_expression

# ==========================================================================================
# Prediction phase
def merge_tensors(D, A, B):
    # Make sure that A and B are two-dimensional
    if A.dim() == 1:
        A = A.unsqueeze(1)
    if B.dim() == 1:
        B = B.unsqueeze(1)
    a_rows, a_cols = A.shape
    b_rows, b_cols = B.shape
    # Ensure that D has at least enough space to store A and B
    d_rows, d_cols = D.shape
    if d_rows < a_rows or d_cols < a_cols + b_cols:
        raise ValueError("Tensor D is not large enough to hold the merged tensors A and B.")
    # Assign A to the left of D
    D[:a_rows, :a_cols] = A
    # Assign B to the right of D
    D[:b_rows, -b_cols:] = B
    return D

sample = torch.zeros(50, 4)
N_sample_point = 50
#################### Specifying test data ####################
X = np.linspace(-4, 4, num=N_sample_point)
X= X.reshape(N_sample_point,1)
# X = abs(X)
x1 = X[:,0]
y = np.sin(x1**2) + x1
#################### Specifying test data ####################

enc_inputs = merge_tensors(sample, torch.tensor(x1), torch.tensor(y)).unsqueeze(0).to(device)

print("=" * 53)
print("Mathematical formula predicted by trained formulaGPT:")

for i in range(len(enc_inputs)):
    for kk in range(1):
        greedy_dec_predict, Best_r2, Best_expression = greedy_decoder(model, enc_inputs[i].unsqueeze(0).to(device), start_symbol=tgt_vocab["S"])
        # greedy_dec_predict = beam_search_decoder(model, enc_inputs[i].unsqueeze(0).to(device), 4, 90, tgt_vocab["S"], tgt_vocab["E"])
        print('################### The Best Expression ###################')
        print('The best $R^2$:', Best_r2)
        print('The best expression: ', Best_expression)


