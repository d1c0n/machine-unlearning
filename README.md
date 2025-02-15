# A MoE Approach to Machine Unlearning

## Overview
This repository contains the implementation of the Deep Learning and Applied AI 2024 project: *A Mixture of Experts (MoE) Approach to Machine Unlearning*. The project proposes an efficient unlearning mechanism by leveraging the modular nature of MoE architectures combined with ideas from the SISA framework.

## Table of Contents
- [Introduction](#introduction)
- [Methodology](#methodology)
- [Results](#results)
- [Installation](#installation)
- [Future Work](#future-work)
- [References](#references)

## Introduction
Machine unlearning is the process of selectively removing specific data from a trained model to address privacy, security, and regulatory concerns. Traditional methods often require retraining from scratch, which is computationally expensive. This project introduces a novel approach that:
- Utilizes a Mixture of Experts (MoE) model to distribute learning across specialized sub-networks.
- Identifies and selectively retrains only the most relevant model components to facilitate efficient unlearning.
- Employs a targeted unlearning loss function to reinforce forgetting.

## Methodology
The method combines principles from the SISA framework and the MoE model:
1. **MoE Layer**: Uses multiple convolutional expert networks, each specializing in different feature sets.
2. **Sparsity Mechanism**: A router network selects a subset of experts to process each input, optimizing computational efficiency.
3. **Targeted Expert Identification**: When a request to forget a specific class is received, the most frequently activated experts for that class are identified.
4. **Selective Retraining**: Instead of retraining the full model, only the identified experts and router network are updated.
5. **Custom Unlearning Loss Function**: Encourages the model to forget target class information by penalizing correct classifications.

## Results
Experiments were conducted using the CIFAR-10 and CIFAR-100 datasets. Key findings include:
- **Optimal Hyperparameters**:
  - Training Data Used: 100%
  - Retraining Epochs: 5
  - Learning Rate: 0.0001
- **Performance**:
  - Accuracy on the forgotten class reduced to 0.7%.
  - Accuracy on other classes retained at >98%.

## Installation
To set up the environment, run:
```bash
pip install -r requirements.txt
```
Ensure you have PyTorch and the necessary dependencies installed.



## Future Work
- Extend to transformer-based architectures.
- Develop more granular unlearning techniques at the feature level.
- Establish theoretical guarantees for selective unlearning.

## References
- Bourtoule et al., 2020: *Machine Unlearning* ([arXiv:1912.03817](https://arxiv.org/abs/1912.03817))
- Jiang et al., 2024: *Mixtral of Experts* ([arXiv:2401.04088](https://arxiv.org/abs/2401.04088))
- Krizhevsky: *The CIFAR-10 and CIFAR-100 Datasets* ([Link](https://www.cs.toronto.edu/~kriz/cifar.html))

