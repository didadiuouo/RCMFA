# RCMFA: Multi-scale Fusion Attention Network Integrating Radiomic and Clinical Features for Prediction of Breast Cancer HER2 Status Changes

This repository contains the official implementation of RCMFA, a deep learning framework designed for radiomic feature learning, cross-modal representation alignment, and downstream prediction tasks (e.g., HER2 status change prediction). The project includes model training, testing, pretrained weights, SHAP visualization utilities, and statistical significance evaluation scripts.

## Environment Setup

Recommended:
Python >= 3.8
PyTorch >= 1.10
CUDA >= 11.3 (optional)

## Training

To train the RCMFA model:

python RCMFA.py --mode train

## Testing & Evaluation

Evaluate a trained model:

python RCMFA.py --mode test --checkpoint ./checkpoints/checkpoint/model_best.pth

## ROC AUC Statistical Comparison (DeLong Test)

Use DeLong's test to compare AUCs between two models:

python Delong_test.py --pred1 modelA_pred.npy --pred2 modelB_pred.npy --label label.npy


## SHAP Interpretability

Generate SHAP plots for model interpretation:

python draw_shap.py --input features.npy --model ./checkpoints/checkpoint/model_best.pth


## Pretrained Weights
Pretrained backbone weights (e.g., PVT) are stored in:
pretrained_pth/pvt/


