# ChestMNIST CNN - Running Instructions

A secure multi-party computation (MPC) implementation of a Convolutional Neural Network for the ChestMNIST medical imaging dataset.

## Overview

This implementation trains a CNN on chest X-ray images using secure MPC protocols. The architecture uses strided convolutions (no MaxPooling) for faster training and binary cross-entropy loss for multi-label classification.

### Architecture

```
Input (N, 28, 28, 1)
  ↓
Conv2D(8, 3×3, stride=2, ReLU)  → (N, 13, 13, 8)
  ↓
Conv2D(16, 3×3, stride=2, ReLU) → (N, 6, 6, 16)
  ↓
Flatten                          → (N, 576)
  ↓
Dense(64, ReLU)
  ↓
Dropout(0.2)
  ↓
Dense(14, linear)                → Multi-label logits
```

**Total Parameters:** ~38,000
**Loss Function:** Binary Cross-Entropy (labels in {0, 1})
**Multi-label Classification:** 14 disease categories

## Prerequisites

### 1. Install Codon

```bash
mkdir $HOME/.codon
curl -L https://github.com/exaloop/codon/releases/download/v0.17.0/codon-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon --strip-components=1
```

### 2. Install Sequre Plugin

```bash
curl -L https://github.com/0xTCG/sequre/releases/download/v0.0.20-alpha/sequre-$(uname -s | awk '{print tolower($0)}')-$(uname -m).tar.gz | tar zxvf - -C $HOME/.codon/lib/codon/plugins
```

### 3. Clone Repository

```bash
git clone https://github.com/0xTCG/sequre.git
cd sequre
```

## Preparing the Dataset

Run the preparation script to generate text files from ChestMNIST:

```bash
python applications/offline/chestmnist_prep.py
```

## Running the Code

### Method 1: Using the Helper Script (Recommended)

```bash
bash scripts/sequre-test.sh applications/offline/chestmnist_cnn.codon
```

### Method 2: Direct Command

From the repository root:

```bash
CODON_DEBUG=lt $HOME/.codon/bin/codon run \
    --disable-opt='core-pythonic-list-addition-opt' \
    -plugin sequre \
    applications/offline/chestmnist_cnn.codon \
    --skip-mhe-setup
```

### Method 3: Release Mode (Faster Performance)

For production runs with better performance (without debug features):

```bash
CODON_DEBUG=lt $HOME/.codon/bin/codon run -release \
    --disable-opt='core-pythonic-list-addition-opt' \
    -plugin sequre \
    applications/offline/chestmnist_cnn.codon \
    --skip-mhe-setup
```

## Configuration

You can modify training parameters by editing the configuration section in `chestmnist_cnn.codon`:

```python
# -- Configuration --
N_TRAIN    = 10000    # Number of training samples
N_TEST     = 3000     # Number of test samples
BATCH_SIZE = 512      # Mini-batch size
EPOCHS     = 5        # Number of training epochs
LR         = 0.001    # Learning rate
MOMENTUM   = 0.9      # Nesterov momentum
```

## Expected Output

The program will display:

1. **Data Loading:**
   ```
   Loading ChestMNIST data ...
     Train  X=(10000, 28, 28, 1)  y=(10000, 14)
     Test   X=(3000, 28, 28, 1)   y=(3000, 14)
   ```

2. **Architecture Summary:**
   ```
   Architecture (strided conv - no MaxPooling)
     Input  -> (N, 28, 28, 1)
     Conv2D(8, 3x3, stride=2, relu)  -> (N, 13, 13, 8)
     Conv2D(16, 3x3, stride=2, relu) -> (N, 6, 6, 16)
     Flatten                          -> (N, 576)
     Dense(64, relu)
     Dropout(0.2)
     Dense(14, linear)
     Loss: binary_crossentropy (mini-batch)
   ```

3. **Training Progress:**
   ```
   Training: 5 epochs, lr=0.001, momentum=0.9, batch_size=512, batches/epoch=20
   ------------------------------------------------------------
     Epoch 1/5  bce_loss = 7.234567  (20 batches)
     Epoch 2/5  bce_loss = 6.123456  (20 batches)
     ...
   ```

4. **Final Results:**
   ```
   ============================================================
   Results
   ============================================================
     Test   accuracy: 0.7234   bce-loss: 5.678901
     Config: 10000 train, 3000 test, batch_size=512, epochs=5
   ```

## Typical Training Time

- **Debug Mode:** ~30-60 minutes (5 epochs, 10K training samples)
- **Release Mode:** ~10-20 minutes
- Actual time depends on hardware (CPU cores, memory)

## Troubleshooting

### Issue: "Cannot import Conv2D"
**Solution:** Ensure the Sequre plugin is properly installed and the neural network layers are synced:
```bash
# Check plugin installation
ls -la $HOME/.codon/lib/codon/plugins/sequre/stdlib/sequre/stdlib/learn/neural_net/
```

### Issue: "File not found: data/chestmnist/*.txt"
**Solution:** Verify data files exist and are in the correct location:
```bash
ls -la data/chestmnist/
```

### Issue: Socket errors
**Solution:** Clean up socket files before running:
```bash
rm -f sock.*
```

## Performance Notes

- **MPC Overhead:** Secure computation is inherently slower than plaintext operations
- **Batch Size:** Larger batches = fewer MPC rounds but more memory
- **Strided Convolutions:** Used instead of MaxPooling for faster MPC training
- **Dropout Rate:** Set to 0.2 (training) and automatically disabled during evaluation

## Architecture Details

### Why Strided Convolutions?

This implementation uses strided convolutions instead of MaxPooling layers because:
- **Fewer MPC operations:** Pooling requires secure comparison operations
- **Faster training:** Reduces computational overhead in secure computation
- **Similar accuracy:** Striding can achieve comparable spatial reduction

### Binary Cross-Entropy Loss

Multi-label classification where each of 14 labels is independently predicted:
- Uses Chebyshev polynomial approximation for secure log computation
- Clips predictions to interval (0.001, 0.999) for numerical stability
- Each label is treated as a separate binary classification problem

## File Structure

Required files for running this code:

```
sequre/
├── applications/offline/chestmnist_cnn.codon  # Main training script
├── data/chestmnist/                            # Dataset files
│   ├── train_images.txt
│   ├── train_labels.txt
│   ├── test_images.txt
│   └── test_labels.txt
├── stdlib/sequre/stdlib/learn/neural_net/     # Neural network library
│   ├── layers.codon                            # Conv2D, Dense, Flatten, Dropout
│   ├── activations.codon                       # ReLU, linear activations
│   └── loss.codon                              # Loss functions
├── stdlib/sequre/stdlib/
│   ├── builtin.codon                           # Secure operations (clip, etc.)
│   └── chebyshev.codon                         # Polynomial approximations
└── scripts/sequre-test.sh                      # Helper script
```

## Citation

If you use this code in your research, please cite the Sequre framework:

```bibtex
@software{sequre2024,
  title={Sequre: A Framework for Secure Multi-Party Computation},
  author={0xTCG},
  year={2024},
  url={https://github.com/0xTCG/sequre}
}
```

## License

See [LICENSE.md](../../LICENSE.md) in the repository root.

## Support

For issues and questions:
- GitHub Issues: https://github.com/0xTCG/sequre/issues
- Repository: https://github.com/0xTCG/sequre
