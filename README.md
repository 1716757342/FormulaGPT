# Generative Pre-Trained Transformer for Symbolic Regression Base In-Context Reinforcement Learning



The repository provides a schematic code of formulaGPT, containing the basic code implementation of formulaGPT. FormulaGPT is trained on extensive sparse reward learning histories from reinforcement learning-based SR algorithms. After training, the SR algorithm based on reinforcement learning is distilled into a Transformer. When new test data comes, FormulaGPT can directly generate a "reinforcement learning process" and automatically update the learning policy in context. Tested on more than ten datasets including SRBench, formulaGPT achieves the state-of-the-art performance in fitting ability compared with four baselines. In addition, it achieves satisfactory results in noise robustness, versatility, and inference efficiency.

# Installation

### Installation - core package

The core package has been tested on Python3.6+ on Unix and OSX. To install the core package, we highly recommend first creating a Python 3 virtual environment, e.g.

```
python3 -m venv venv3 # Create a Python 3 virtual environment
source venv3/bin/activate # Activate the virtual environment
```
Then, from the repository root:
```
pip install -r requirements.txt # Install packages and core dependencies
```

# Start training

### Step 1: `save_pth `specifies the save path for the model. `train_data_filename` is the path to the training data. 

```
save_pth = 'formulaGPT-epoch-ex.pth' ### 指定模型保存路径
train_data_filename = "data_symbols.json" #### 指定训练数据路径
```

### Step 2: Specifies the relevant hyperparameters of formulaGPT
```
epochs = 400 
src_len = 8  # enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length

# Transformer Parameters
d_model = 512  # Embedding Size
# FeedForward dimension 
d_ff = 2048
d_k = d_v = 64   # dimension of K(=Q), V
n_layers = 6 # number of Encoder of Decoder Layer
n_heads = 8  # number of heads in Multi-Head Attention
```
### Step 3: Run `formulaGPT_train.py` to train the model
```
python formulaGPT_train.py
```
# Start inference
### Step1: Specifies the model loading path
Start by specifying the path to load the model via `model_pth`.
```
model_pth = 'formulaGPT-epoch-10000.pth' #### 指定要加载的模型路径
```

### Step2: Specify the model hyperparameters
Make sure the following hyperparameters are the same as when training the model
```
src_len = 8  # enc_input max sequence length
tgt_len = 16  # dec_input(=dec_output) max sequence length
# formulaGPT-1000
# Transformer Parameters
d_model = 512  # Embedding Size（token embedding和position编码的维度）
# FeedForward dimension (两次线性层中的隐藏层 512->2048->512，线性层是用来做特征提取的），当然最后会再接一个projection层
d_ff = 2048
d_k = d_v = 252*1  # dimension of K(=Q), V（Q和K的维度需要相同，这里为了方便让K=V）
n_layers = 8 # number of Encoder of Decoder Layer（Block的个数）
n_heads = 8 * 2  # number of heads in Multi-Head Attention（有几套头）
```
### Step3: Given the test data `[X,y]`
Here `[X,y]` can be input from the outside world or generated randomly.
```
#################### 指定测试数据 ####################
X = np.linspace(-4, 4, num=N_sample_point)
X= X.reshape(N_sample_point,1)
x1 = X[:,0]
y = np.sin(x1**2) + x1
#################### 指定测试数据 ####################
```
### Step3: Run `formuGPT_test.py`
Run the file `formuGPT_test.py` with the following command:
```
python formuGPT_test.py
```
If it runs successfully you should get output similar to the following
```

################### The Best Expression ###################
The best $R^2$: 0.9999999999999951
The best expression:  ['+', 'sin', '*', 'var_x1', 'var_x1', 'var_x1']

```

