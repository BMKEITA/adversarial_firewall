# Multi-Class Adversarial Firewall: A Lightweight Defense Framework for Real-Time ML Systems

 In this work, we introduce a lightweight, modular firewall layer for upstream adversarial attack classification and defense selection.

## Features
- **Multi-Class Attack Detection**: Classifies FGSM, PGD, Noise, MIA, CW, DeepFool, and SPSA attacks
- **Adaptive Defense**: Automatically selects appropriate defense strategy for each attack type
- **Real-Time Performance**: Optimized for <20ms inference time
- **Enhanced Feature Extraction**: Spatial and frequency domain features
- **Modular Architecture**: Easy to integrate with existing models

## Architecture
- **EnhancedFirewallNetwork**: Neural network for attack classification
- **EnhancedSpatialExtractor**: Extracts spatial features (edges, textures, gradients)
- **EnhancedFrequencyExtractor**: Extracts frequency domain features (DCT coefficients)
- **DefenseModule**: Implements various defense strategies
- **AdversarialFirewall**: Main firewall class with training and deployment capabilities

## Installation

```bash
git clone https://github.com/BMKEITA/adversarial_firewall.git
cd adversarial_firewall

## Install dependencies
pip install -r requirements.txt

## Basic training (5 classes):
python adversarial_firewall.py --mode train --dataset CIFAR10 --epochs 25

## Training with all attack types:
python adversarial_firewall.py --mode train --dataset CIFAR10 --include_all_attacks

## Model compression:
python adversarial_firewall.py --mode compress --load firewall_output/best_firewall_model.pth --compress_ratio 0.3

## Full demonstration:
python adversarial_firewall.py --mode demo --dataset CIFAR10 --num_classes 5

