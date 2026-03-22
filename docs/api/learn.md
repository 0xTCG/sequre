# Secure Machine Learning

_Defined in `stdlib/sequre/stdlib/learn/`_

Sequre provides privacy-preserving implementations of common machine learning algorithms. All models are generic over their data type `T`, so the same code works with `Sharetensor`, `MPU`, or plain `ndarray`.

---

## Linear regression — `LinReg[T]`

_Defined in `stdlib/sequre/stdlib/learn/lin_reg.codon`_

Secure linear regression with bias, supporting batch gradient descent (BGD), mini-batch gradient descent (MBGD), and closed-form solutions for small feature counts.

### Construction

```python
from sequre.stdlib.learn.lin_reg import LinReg

model = LinReg[MPU](mpc)                             # empty weights
model = LinReg(initial_weights, optimizer="bgd")      # with initial weights
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `fit` | `fit(mpc, X, y, step, epochs, verbose=False)` | Train the model. Returns `self`. |
| `predict` | `predict(mpc, X, noise_scale=0.0)` | Predict on new data. Optional DP noise. |
| `loss` | `loss(mpc, X, y)` | Compute MSE loss $\|y - X\beta\|^2$ |
| `randomize_weights` | `randomize_weights(mpc, distribution="uniform")` | Reinitialize weights randomly |
| `estimate_step` | `LinReg.estimate_step(train, test)` | Static: estimate learning rate from covariance |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `coef_` | `T` | Learned weight vector (including bias) |
| `optimizer` | `str` | `"bgd"` (batch), `"mbgd"` (mini-batch), or `""` (auto: closed-form if features < 4) |

### Optimizers

| Optimizer | Description |
|---|---|
| `"bgd"` | Batch gradient descent: pre-computes $X^TX$ and $X^Ty$, then iterates $w \leftarrow w + \eta(X^Ty - X^TXw)$ |
| `"mbgd"` | Mini-batch gradient descent: splits data into 10 batches, same update rule per batch |
| (auto) | If feature count < 4, uses closed-form: $w = (X^TX)^{-1}X^Ty$ |

---

## Logistic regression — `LogReg[T]`

_Defined in `stdlib/sequre/stdlib/learn/log_reg.codon`_

Secure logistic regression supporting binary (sigmoid) and multinomial (softmax) classification via Chebyshev-approximated activation functions.

### Construction

```python
from sequre.stdlib.learn.log_reg import LogReg

model = LogReg(initial_weights, optimizer="bgd", interval=(-50.0, 10.0), variant="binary")
model = LogReg(initial_weights, variant="multinomial", interval=(-20.0, 0.0))
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `fit` | `fit(mpc, X, y, step, epochs, verbose=False)` | Train the model. Returns `self`. |
| `predict` | `predict(mpc, X)` | Predict class probabilities |
| `loss` | `loss(mpc, X, y)` | Compute cross-entropy loss |
| `randomize_weights` | `randomize_weights(mpc, distribution="uniform")` | Reinitialize weights randomly |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `coef_` | `T` | Learned weight matrix |
| `optimizer` | `str` | `"bgd"` or `"mbgd"` |
| `variant` | `str` | `"binary"` (sigmoid activation) or `"multinomial"` (softmax activation) |
| `interval` | `tuple[float, float]` | Chebyshev approximation interval for the activation function |

### Activation functions

| Variant | Activation | Chebyshev approximation |
|---|---|---|
| `"binary"` | $\sigma(z) = \frac{1}{1+e^{-z}}$ | `chebyshev_sigmoid` with clipping |
| `"multinomial"` | $\text{softmax}(z) = \frac{e^{z_i - \max(z)}}{\sum e^{z_j - \max(z)}}$ | `chebyshev_exp` with shift and clipping |

---

## Linear SVM

_Defined in `stdlib/sequre/stdlib/learn/lin_svm.codon`_

Secure linear support-vector machine with hinge loss. Provides both offline (plaintext) and secure (MPC) implementations.

### Secure functions

| Function | Signature | Description |
|---|---|---|
| `backprop` | `backprop(mpc, x, y, w, b, l2)` | Single-sample gradient computation |
| `lsvm_predict` | `lsvm_predict(mpc, X, w, b)` | Predict: $Xw - b$ |
| `lsvm_score` | `lsvm_score(mpc, X, Y, w, b, l2)` | Hinge loss + L2 regularization |
| `lsvm_train` | `lsvm_train(mpc, X, Y, eta, epochs, l2, mini_batch_size, optimizer)` | Full training loop |

### Optimizers

| Optimizer | Description |
|---|---|
| `"sgd"` | Stochastic gradient descent (single-sample updates) |
| `"bgd"` | Batch gradient descent (full-dataset gradient) |
| `"mbgd"` | Mini-batch gradient descent |

### Offline (plaintext) counterparts

For benchmarking and data preparation: `offline_backprop`, `offline_lsvm_predict`, `offline_lsvm_score`, `offline_lsvm_train`.

---

## Principal component analysis (PCA)

_Defined in `stdlib/sequre/stdlib/learn/pca.codon`_

Randomized PCA for dimensionality reduction, designed for secure distributed settings using the sketch-and-solve paradigm.

### Helper classes

#### `RandomSketching`

| Method | Description |
|---|---|
| `iterative(mpc, data, miss, data_mean, top_k, oversampling)` | Build a sketch from data with missing values via random bucketing |
| `vectorized(mpc, data, sketch_size)` | Build a sketch using random matrix multiplication |
| `generate_sketch_matrix(shape)` | Generate a random sketching matrix (plaintext) |

#### `PowersStep`

| Method | Description |
|---|---|
| `with_lazy_norm(mpc, sketch, data, miss, mean, std_inv, iters)` | Power iteration with lazy normalization (handles missing data) |
| `without_norm(mpc, sketch, data, iters)` | Power iteration without normalization |

### Top-level functions

| Function | Description |
|---|---|
| `random_pca_with_norm(mpc, data, miss, mean, std_inv, top_k, oversampling, power_iters, filtered_size)` | Full PCA pipeline with normalization and missing-data handling |
| `random_pca_without_norm(mpc, data, mean, top_k, oversampling, power_iters)` | PCA without normalization; uses `via_mpc` for eigen decomposition |
| `random_pca_without_projection(mpc, data_mpp, top_k, oversampling, power_iters)` | PCA on MPP data (Algorithm 1 from [arXiv:2304.00129](https://arxiv.org/abs/2304.00129)); includes distributed QR via `via_mpc` |

---

## Multiple imputation (MI)

_Defined in `stdlib/sequre/stdlib/learn/mi.codon`_

Secure multiple imputation for handling missing data using Rubin's rules.

### `Imputer[M]`

A wrapper around any regression model `M` (e.g., `LinReg`) that handles train/test splitting around missing values and imputation.

| Method | Description |
|---|---|
| `Imputer(model)` | Wrap a regression model |
| `fit(mpc, complete_data, labels, step, epochs, mode)` | Train the underlying model on complete cases |
| `impute(mpc, data, miss_rows, miss_col, step, epochs, noise_scale)` | Split, train, and impute missing values |
| `impute_inplace(mpc, data, mask, miss_col, noise_scale)` | Impute missing values in-place using a boolean mask |

Static helpers:

| Method | Description |
|---|---|
| `Imputer.count_missing_data(data, target_col, miss_val)` | Count missing entries in a column |
| `Imputer.split_train_test(data, target_col, ...)` | Split into complete and incomplete rows. Works on both `ndarray` and `MPU`. |

### `MI[IM, FM]`

Multiple imputation with Rubin's combining rules. `IM` is the imputation model type, `FM` is the final analysis model type.

| Method | Description |
|---|---|
| `MI(factor, impute_model, fit_model)` | Create MI with `factor` imputations |
| `fit(mpc, data, labels, miss_rows, miss_col, ...)` | Run full MI pipeline: impute `factor` times, fit `FM` on each, combine via Rubin's rules |
| `MI.rubin(mpc, weights)` | Static: combine weight estimates using Rubin's combining rules |

### Imputation modes

| Mode | Constant | Description |
|---|---|---|
| Batched | `MI_BATCHED_MODE` | Train imputer once, generate `factor` imputations with noise |
| Stochastic | `MI_STOCHASTIC_MODE` | Retrain imputer with random weights for each imputation |

### `MICE[IM, FM]`

Multiple Imputation by Chained Equations — extends `MI` for datasets with multiple columns containing missing values. Iteratively imputes each column conditioned on the others.

---

## Neural networks — `Sequential[L]`

_Defined in `stdlib/sequre/stdlib/learn/neural_net/model.codon`_

A Keras-style sequential neural network supporting secure training with batch and mini-batch gradient descent and Nesterov momentum.

### Construction

```python
from sequre.stdlib.learn.neural_net.model import Sequential
from sequre.stdlib.learn.neural_net.layers import Input, Dense
from sequre.types.multiparty_union import MPU

layers = (
    Input[MPU](16),
    Dense[MPU]("relu", 32, "normal", "zeros"),
    Dense[MPU]("linear", 1, "normal", "zeros"))

model = Sequential(layers).compile(mpc, loss="hinge", optimizer="bgd")
```

### Methods

| Method | Signature | Description |
|---|---|---|
| `compile` | `compile(mpc, loss, optimizer, *args, **kwargs)` | Initialize layer weights and set loss/optimizer. Returns `self`. |
| `fit` | `fit(mpc, X, y, step, epochs, momentum, batch_size=0, verbose=False)` | Train the model. `batch_size` is required when `optimizer="mbgd"`. |
| `predict` | `predict(mpc, X)` | Forward pass; returns the output of the last layer. |
| `get_loss` | `get_loss(mpc, X, y)` | Forward pass + loss computation. |

### Optimizers

| Optimizer | Description |
|---|---|
| `"bgd"` | Batch gradient descent: full-dataset gradient each epoch |
| `"mbgd"` | Mini-batch gradient descent: splits data into batches of `batch_size` |

### Layers

_Defined in `stdlib/sequre/stdlib/learn/neural_net/layers.codon`_

#### `Input[ctype]`

The input layer. Holds no trainable parameters — simply passes through data.

```python
Input[type(X)](size)
```

| Parameter | Type | Description |
|---|---|---|
| `size` | `int` | Number of input features |

#### `Dense[ctype]`

A fully-connected layer with configurable activation and weight initialization.

```python
Dense[type(X)](activation, size, kernel_initializer="uniform", bias_initializer="uniform")
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `activation` | `str` | — | Activation function: `"relu"` or `"linear"` |
| `size` | `int` | — | Number of neurons |
| `kernel_initializer` | `str` | `"uniform"` | Weight initializer: `"uniform"`, `"normal"`, `"zeros"`, or `"ones"` |
| `bias_initializer` | `str` | `"uniform"` | Bias initializer (same options) |

Weight updates use **Nesterov accelerated gradient** with the `momentum` parameter passed to `fit`.

### Activations

_Defined in `stdlib/sequre/stdlib/learn/neural_net/activations.codon`_

| Activation | Function | Derivative |
|---|---|---|
| `"relu"` | $\text{ReLU}(x) = x \cdot \mathbf{1}[x > 0]$ | $\mathbf{1}[x > 0]$ |
| `"linear"` | $f(x) = x$ | $1$ |

### Losses

_Defined in `stdlib/sequre/stdlib/learn/neural_net/loss.codon`_

| Loss | Function | Derivative |
|---|---|---|
| `"hinge"` | $\frac{1}{n}\sum \max(0,\; 1 - y \hat{y})$ | $\frac{1}{n}(-y \cdot \mathbf{1}[1 - y\hat{y} > 0])$ |

### Examples

#### Credit score classification — [applications/credit_score.codon](https://github.com/0xTCG/sequre/blob/develop/applications/credit_score.codon)

Binary credit-score prediction with a single hidden layer. Trains on secret-shared data, then evaluates accuracy/precision/recall/F1.

```python
from sequre.stdlib.learn.neural_net.layers import Input, Dense
from sequre.stdlib.learn.neural_net.model import Sequential

layers = (
    Input[type(X)](X.shape[1]),
    Dense[type(X)]("relu", 32, "normal", "zeros"),
    Dense[type(X)]("linear", 1, "normal", "zeros"))

model = Sequential(layers).compile(mpc, loss="hinge", optimizer="bgd")
model.fit(mpc, X=X, y=y, epochs=epochs, step=step_size, momentum=momentum, verbose=verbose)

prediction = model.predict(mpc, X)
```

#### Drug-target interaction — [applications/dti.codon](https://github.com/0xTCG/sequre/blob/develop/applications/dti.codon)

DTI inference with a larger network (128 neurons) over 8192-dimensional feature vectors, using mini-batch gradient descent.

```python
layers = (
    Input[type(X)](8192),
    Dense[type(X)]("relu", 128, "normal", "zeros"),
    Dense[type(X)]("linear", 1, "normal", "zeros"))

model = Sequential(layers).compile(mpc, loss="hinge", optimizer="bgd")
model.fit(mpc, X=X, y=y, step=0.2, epochs=50, momentum=0.9, batch_size=16)
```
