# Generative Pre-Trained Transformer for Symbolic Regression Base In-Context Reinforcement Learning



The repository provides a schematic code of formulaGPT, containing the basic code implementation of formulaGPT. FormulaGPT is trained on extensive sparse reward learning histories from reinforcement learning-based SR algorithms. After training, the SR algorithm based on reinforcement learning is distilled into a Transformer. When new test data comes, FormulaGPT can directly generate a "reinforcement learning process" and automatically update the learning policy in context. Tested on more than ten datasets including SRBench, formulaGPT achieves the state-of-the-art performance in fitting ability compared with four baselines. In addition, it achieves satisfactory results in noise robustness, versatility, and inference efficiency.
<br>

![Python Versions](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)

<br>

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
save_pth = 'formulaGPT-epoch-ex.pth' ### Specifies the model save path
train_data_filename = "train_data.json" #### Specify the training data path
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
model_pth = 'formulaGPT-epoch-10000.pth' #### Specifies the model path to load
```

### Step2: Specify the model hyperparameters
Make sure the following hyperparameters are the same as when training the model
```
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
```
### Step3: Given the test data `[X,y]`
Here `[X,y]` can be input from the outside world or generated randomly.
```
#################### Specifying test data ####################
X = np.linspace(-4, 4, num=N_sample_point)
X= X.reshape(N_sample_point,1)
x1 = X[:,0]
y = np.sin(x1**2) + x1
#################### Specifying test data ####################
```
### Step3: Run `formulaGPT_test.py`
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

