
# ======================================
# === Pytorch hand-written FormulaGPT schematic code
# ======================================

import math

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from set_encoder import SetEncoder
import numpy as np
import copy
import json

save_pth = 'formulaGPT-epoch-ex.pth' ### Specifies the model save path
train_data_filename = "data_symbols.json" #### Specify the training data path
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def pad_sequence(sequence, target_length=40, padding_token='P'):
    # Use whitespace to split the sequence into lists of words
    words = sequence.split(' ')

    # Count the number of words to fill
    num_padding = target_length - len(words)
    # If there are enough words, return only the first target_length words
    if num_padding <= 0:
        return ' '.join(words[:target_length])

    # Otherwise, the padding is added and returned
    padded_words = words + [padding_token] * num_padding
    return ' '.join(padded_words)
def read_from_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading from file: {e}")
        return None

# Examples of Use

### To load data from an offline.json file, use the following code ####
filename = train_data_filename
data = read_from_file(filename)
train_data = data["symbols"]

sentences = []

### Specifies the simulated training data
# train_data = [
#     ['S + * + * sin * var_x1 + var_x1 var_x1 var_x1 var_x1 var_x1 var_x1 1.0 + * + var_x1 * var_x1 var_x1 var_x1 var_x1 1.0 E'],
#     ['S * sin var_x1 var_x1 0.0 * var_x1 var_x1 1.0 E'],
#     ['S cos var_x1 0.0 sin var_x1 1.0 E'],
# ]

## padding
for i in range(len(train_data)):
    pad_s = pad_sequence(train_data[i][0])
    pad_words = pad_s.split(' ')
    dec = [pad_words[0:-1], pad_words[1:]]
    sentences.append(dec)

# Padding Should be Zero
tgt_vocab = {'P': 0, '+': 1, '-':2,'*': 3, '/':4,'sin': 5,'cos':6, 'exp':7,'var_x1': 8,
             '0.1': 9, '0.2': 10, '0.3': 11, '0.4': 12, '0.5': 13,'0.6': 14,'0.7': 15,'0.8': 16,'0.9': 17,'1.0': 18,'0.0': 19, 'S': 20,'E':21}
idx2word = {i: w for i, w in enumerate(tgt_vocab)}
# print('idx2word',idx2word)
tgt_vocab_size = len(tgt_vocab)
# print(tgt_vocab_size)

# transformer epochs
epochs = 40
src_len = 8  # enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 512  # Embedding Size
# FeedForward dimension
d_ff = 2048
d_k = d_v = 64   # dimension of K(=Q), V
n_layers = 6 # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention

# ==============================================================================================
# Data construction
def make_data(sentences):
    """ Converts a sequence of words to a sequence of numbers. """
    dec_inputs, dec_outputs = [], []
    for i in range(len(sentences)):
        # print(sentences)
        # enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][0]]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][1]]]
        # [[1, 2, 3, 4, 5, 6, 7, 0], [1, 2, 8, 4, 9, 6, 7, 0], [1, 2, 3, 4, 10, 6, 7, 0]]
        # enc_inputs.extend(enc_input)
        # [[9, 1, 2, 3, 4, 5, 11], [9, 1, 2, 6, 7, 5, 11], [9, 1, 2, 3, 8, 5, 11]]
        dec_inputs.extend(dec_input)
        # [[1, 2, 3, 4, 5, 11, 10], [1, 2, 6, 7, 5, 11, 10], [1, 2, 3, 8, 5, 11, 10]]
        dec_outputs.extend(dec_output)
        # print(dec_outputs)
    return  torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

dec_inputs, dec_outputs = make_data(sentences)
enc_inputs = torch.randn(3, 20, 4) * 4

enc_inputs[0,:,-1] = enc_inputs[0,:,0] **3 + enc_inputs[0,:,0] **2 + enc_inputs[0,:,0] **1
enc_inputs[1,:,-1] = enc_inputs[1,:,0] **2
enc_inputs[2,:,-1] = torch.sin(enc_inputs[2,:,0])
enc_inputs[:,:,1:-1] = 0

enc_test = copy.deepcopy(enc_inputs)

class MyDataSet(Data.Dataset): # Similar to concatenating data
    """自定义DataLoader"""
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]

    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]

loader = Data.DataLoader(  # batch-train, break the data into small batches to train on.

    MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)


# ====================================================================================================
# Transformer model

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model) ##5000*512
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) ##5000 * 1
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1) #shape = [5000，1，512]
        # print(pe.shape)
        self.register_buffer('pe', pe) ##Tensors registered with register_buffer() : automatically become parameters in the model, move as the model moves (gpu/cpu), but are not updated with gradients.

    def forward(self, x):
        """
        x: [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def get_attn_pad_mask(seq_q, seq_k):
    """Here q,k represents two sequences (not related to q,k of the attention mechanism), such as encoder_inputs (x1,x2,.. xm) and encoder_inputs (x1,x2.. xm)
    Both encoder and decoder may call this function, so seq_len depends
    seq_q: [batch_size, seq_len]
    seq_k: [batch_size, seq_len]
    seq_len could be src_len or it could be tgt_len
    seq_len in seq_q and seq_len in seq_k maybe not equal
    """

    batch_size, len_q = seq_q.size()  # This seq_q is only used to expand the dimension
    batch_size, len_k = seq_k.size()
    # eq(zero) is PAD token
    # e.g. :seq_k = [[1,2,3,4,0], [1,2,3,5,0]]
    # [batch_size, 1, len_k], True is masked
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1) ## seq_k.data.eq(0) Anything equal to 0 in seq_k is true.

    return pad_attn_mask.expand(batch_size, len_q, len_k)

def get_attn_subsequence_mask(seq): ## Only for decoding
    """
    seq: [batch_size, tgt_len]
    """
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)  # Generate an upper triangular matrix
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()
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
        # This fully connected layer ensures that the output of the multi-head attention is still seq_len x d_model
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
        # K: [batch_size, n_heads, len_k, d_k]
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

## Linear in Pytorch only operates on the last dimension, so we want the same fully connected network for every location
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
        # First enc_inputs * W_Q = Q
        # Second enc_inputs * W_K = K
        # Third enc_inputs * W_V = V

        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,
                                               enc_self_attn_mask)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)
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
        return dec_outputs, dec_self_attn, dec_enc_attn
class Config:
    def __init__(self, N_p, activation, bit16, dec_layers, dec_pf_dim, dim_hidden,
                 dim_input,dropout,input_normalization ,length_eq,
    linear,
    ln ,
    lr ,
    mean ,
    n_l_enc ,
    norm ,
    num_features ,
    num_heads ,
    num_inds ,
    output_dim ,
    sinuisodal_embeddings,
    src_pad_idx ,
    std ,
    trg_pad_idx):
        self.N_p= N_p
        self.activation= activation
        self.bit16= bit16
        self.dec_layers= dec_layers
        self.dec_pf_dim= dec_pf_dim
        self.dim_hidden= dim_hidden
        self.dim_input= dim_input ## 输入维度
        self.dropout= dropout
        self.input_normalization= input_normalization
        self.length_eq= length_eq
        self.linear= linear
        self.ln= ln
        self.lr= lr
        self.mean= mean
        self.n_l_enc= n_l_enc
        self.norm= norm
        self.num_features= num_features #### The number of lines printed is num_features * 512
        self.num_heads= num_heads
        self.num_inds= num_inds
        self.output_dim= output_dim
        self.sinuisodal_embeddings= src_pad_idx
        self.src_pad_idx= src_pad_idx
        self.std= std
        self.trg_pad_idx= trg_pad_idx
# Create a configuration object and set its properties
cfg = Config(0, 'relu', True, 5, 512, 512, 4, 0, False, 60, False, True, 0.0001, 0.5, 5, True, 4, 8, 50, 60,
             False, 0, 0.5, 0)
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # self.src_emb = nn.Embedding(src_vocab_size, d_model)  # token Embedding
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
            0, 1)  # [batch_size, src_len, d_model] ### The input sequence plus position encoding is obtained
                # Encoder pad mask matrix of the input sequence

        enc_self_attn_mask = torch.zeros(enc_outputs.size()[0], enc_outputs.size()[1], 4, dtype=torch.bool).cuda()

        enc_self_attns = []
        for layer in self.layers:  # Loop over the nn.ModuleList object
           # The outputs of the previous block enc_outputs are the inputs of the current block
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs,
                                               enc_self_attn_mask)
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
        enc_outputs: [batch_size, src_len, d_model]   # 用在Encoder-Decoder Attention层
        """
        dec_outputs = self.tgt_emb(
            dec_inputs)  # [batch_size, tgt_len, d_model]

        dec_outputs = self.pos_emb(dec_outputs.transpose(0, 1)).transpose(0, 1).to(
            device)  # [batch_size, tgt_len, d_model]
        # Decoder pad mask matrix of the input sequence (decoder is not padded in this example, in practice it is padded)
        dec_self_attn_pad_mask = get_attn_pad_mask(dec_inputs, dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]

        # Masked Self_Attention：There is no future information in the present moment
        dec_self_attn_subsequence_mask = get_attn_subsequence_mask(dec_inputs).to(
            device)  # [batch_size, tgt_len, tgt_len]

        # In the Decoder, the two mask matrices are added together (both the information of the pad and the information of the future time are masked).
        dec_self_attn_mask = torch.gt((dec_self_attn_pad_mask + dec_self_attn_subsequence_mask),
                                      0).to(device)  # [batch_size, tgt_len, tgt_len]; torch.gt The elements of the two matrices are compared, and 1 is returned if the value is greater than, and 0 otherwise

        # This mask is mainly used in the encoder-decoder attention layer

        enc_inputs_mask = torch.ones(enc_inputs.size()[0],enc_inputs.size()[2]).cuda()
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
    if s in ['x','var_x1', 'var_x2', 'var_x3','var_x4','var_x5','var_x6','var_x7','var_x8','var_x9','c']:
        return 0
    if s in ['sin', 'cos', 'exp', 'ln', 'sqrt']:
        return 1
    if s in ['+', '-', '*', '/', '^']:
        return 2
def R2(y,y_pred):
    return 1 - np.mean((y-y_pred)**2)/np.mean((y - np.mean(y))**2)
def process_R2(R2, threshold=1):
    if R2 <= 0 or threshold > 1:
        return str(float(0))
    else:
        return str(float(round(R2, 1)))
def all_farward(l2,X):
    global stack1 ,v1
    stack1 = []
    for i in range(len(l2)):
        s = l2[-(i + 1)]
        if s == 'var_x1':
            stack1.append(X)
        if s == '0':
            mkl = 0
        if s in ['sin', 'cos', 'log', 'exp', 'sqrt']:
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
optimizer = optim.SGD(model.parameters(), lr=1e-4,
                      momentum=0.99)
# ====================================================================================================
for epoch in range(epochs):
    if epoch%40 == 0:
        # The model parameters are saved every 40 epochs and can be adjusted as needed.
        torch.save(model.state_dict(), save_pth)
        print('Model weights saved epoch ' + str(epoch))
    for enc_inputs, dec_inputs, dec_outputs in loader:
        # print('enc_inputs, dec_inputs, dec_outputs',enc_inputs, dec_inputs, dec_outputs)
        """
        enc_inputs: [batch_size, src_len]
        dec_inputs: [batch_size, tgt_len]
        dec_outputs: [batch_size, tgt_len]
        """
        # print('dec_outputs-11', dec_outputs.view(-1))
        enc_inputs, dec_inputs, dec_outputs = enc_inputs.to(
            device), dec_inputs.to(device), dec_outputs.to(device)
        # outputs: [batch_size * tgt_len, tgt_vocab_size]
        outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(
            enc_inputs, dec_inputs)

        loss = criterion(outputs, dec_outputs.view(-1)) ## Here all labels of each batch are flattened like [2,7] ->[1,14]
        print('Epoch:', '%04d' % (epoch + 1), 'loss =', '{:.6f}'.format(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

