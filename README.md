# FormulaGPT: A Transformer-based Model for Symbolic Regression

<p align="center">
    <img src="FormulaGPT_v2.png" alt="FormulaGPT Logo" width="200"/>
</p>

[![Python Version](https://img.shields.io/badge/Python-3.7%20%7C%203.8%20%7C%203.9-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue)](https://opensource.org/licenses/Apache-2.0)
[![GitHub Stars](https://img.shields.io/github/stars/username/repo.svg?style=social)](https://github.com/username/repo)
[![GitHub Forks](https://img.shields.io/github/forks/username/repo.svg?style=social)](https://github.com/username/repo)

---

## Overview

FormulaGPT is a cutting-edge Transformer-based model designed for **Symbolic Regression (SR)**. It leverages extensive learning histories from reinforcement learning-based SR algorithms to generate mathematical formulas directly from data points. After training, FormulaGPT can automatically update its learning policy in context, making it highly versatile for diverse regression tasks.

<p align="center">
    <img src="images/model_overview.png" alt="Model Overview" width="800"/>
</p>

## Features

- **State-of-the-art Performance**: Achieves superior fitting ability compared to traditional SR methods.
- **Noise Robustness**: Demonstrates excellent resilience against noisy datasets.
- **Versatility**: Effectively handles a wide range of regression problems.
- **Efficient Inference**: Optimized for quick formula generation during inference.

## Quick Installation

### Prerequisites

- Python 3.7 or higher
- PyTorch 1.7 or higher
- Basic knowledge of deep learning and symbolic regression

### Installation Steps

```bash
# Clone this repository
git clone https://github.com/username/formulaGPT.git
cd formulaGPT

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Getting Started

### 1. Start Training

#### Configuration

```python
# Example configuration for training
save_pth = 'formulaGPT-epoch-ex.pth'  # Path to save the trained model
train_data_filename = "train_data.json"  # Path to your training data

# Hyperparameters
epochs = 400
src_len = 8  # Maximum sequence length for encoder input
tgt_len = 16  # Maximum sequence length for decoder input/output

# Transformer Parameters
d_model = 512  # Embedding size
d_ff = 2048  # FeedForward dimension
d_k = d_v = 64  # Dimension of K and V
n_layers = 6  # Number of Encoder/Decoder layers
n_heads = 8  # Number of attention heads
```

#### Run Training

```bash
python formulaGPT_train.py
```

---

### 2. Start Inference

#### Configuration

```python
# Example configuration for inference
model_pth = 'formulaGPT-epoch-10000.pth'  # Path to your trained model

# Ensure hyperparameters match those used during training
src_len = 8
tgt_len = 16
d_model = 512
d_ff = 2048
d_k = d_v = 252
n_layers = 8
n_heads = 16
```

#### Example Test Data

```python
import numpy as np

N_sample_point = 50
X = np.linspace(-4, 4, num=N_sample_point).reshape(N_sample_point, 1)
x1 = X[:, 0]
y = np.sin(x1 ** 2) + x1
```

#### Run Inference

```bash
python formulaGPT_test.py
```

---

## Results

When tested on datasets like SRBench, FormulaGPT achieves state-of-the-art performance. Here’s an example output:

```
################### The Best Expression ###################
The best $R^2$: 0.9999999999999951
The best expression:  ['+', 'sin', '*', 'var_x1', 'var_x1', 'var_x1']
```

---

## Why FormulaGPT?

- **Efficiency**: The model is optimized for quick inference, making it suitable for real-time applications.
- **Versatility**: Works across various domains, from physics to finance.
- **Interpretability**: Generates human-readable mathematical expressions, enhancing trust and understanding.

---

## Contributing

Contributions are welcome! If you have any suggestions or improvements for FormulaGPT, please fork the repository and submit a pull request.

---

## Contact

For any questions or feedback, feel free to reach out at [your.email@example.com](mailto:your.email@example.com).

---

## License

This project is [Apache License 2.0](LICENSE)-licensed. Feel free to use, modify, and distribute this code as per the terms of the license.

---

Thank you for choosing FormulaGPT! We hope this tool enhances your symbolic regression tasks and look forward to seeing the innovative applications you build with it!
