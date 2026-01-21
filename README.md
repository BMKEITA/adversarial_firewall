# Multi-Class Adversarial Firewall: Attack-Type Classification for Adaptive Defense in Visual Computing Systems

The susceptibility of machine learning models to adversarial attacks presents serious risks in safety-critical domains. We present in this paper a Multi-Class Adversarial Firewall, a lightweight defense framework that detects and classifies multiple adversarial attacks in real-time while applying targeted countermeasures. In contrast to current detection techniques that carry out binary classification or concentrate on single attack types, our approach distinguishes between clean samples and specific attack classes including FGSM, PGD, Noise, and Membership Inference Attacks (MIA). The framework incorporates both spatial (Laplacian, edge density, local entropy) and frequency domain (DCT coefficients, spectral energy distribution) features within a compact neural network to apply the particular  defense strategy based on confidence. 
	Experimental evaluation shows 95.07\% overall accuracy with 94.70\% detection rate on MNIST, 95.00\% accuracy with 99.25\% detection rate on CIFAR-10, and 86.60\% accuracy with 99.38\% detection rate on CIFAR-100, while maintaining a model size of 0.20 MB and average inference time of 35-44 ms across datasets. Our system significantly outperforms existing binary detection methods and provides attack-specific defense strategies, enabling practical deployment in resource-constrained environments.

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

# Installation
```bash
# Clone the repository
git clone https://github.com/BMKEITA/adversarial_firewall.git
cd adversarial_firewall

# Install dependencies
pip install -r requirements.txt

# Run demo (5 classes):
python adversarial_firewall.py --mode demo --dataset CIFAR10

# Training with all attack types:
python adversarial_firewall.py --mode train --dataset CIFAR10 --include_all_attacks

# Model compression:
python adversarial_firewall.py --mode compress --load firewall_output/best_firewall_model.pth --compress_ratio 0.3

# Full demonstration:
python adversarial_firewall.py --mode demo --dataset CIFAR10 --num_classes 5
```
#Project Structure
```bash
adversarial_firewall/
├── adversarial_firewall.py    # Main implementation file
├── requirements.txt           # Python dependencies
├── firewall_output/           # Generated outputs (checkpoints, logs, visualizations)
│   ├── best_firewall_model.pth
│   ├── training_history.png
│   ├── evaluation_results.json
│   └── visual_examples/
│       ├── attack_comparison.png
│       ├── spatial_features.png
│       └── frequency_features.png
└── README.md                  # This file
```


