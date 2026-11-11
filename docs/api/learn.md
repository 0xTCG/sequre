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
| `predict` | `predict(mpc, X, noise_scale=0.0)` | Predict on new data. With `noise_scale`, each prediction gets `noise_scale` times standard normal noise (the noise of stochastic imputation in [MI](#multiple-imputation-mi); not a privacy mechanism). |
| `loss` | `loss(mpc, X, y)` | Compute MSE loss $\|y - X\beta\|^2$ |
| `randomize_weights` | `randomize_weights(mpc, distribution="uniform")` | Reinitialize weights randomly (public randomness: for a `Sharetensor` model CP1 draws the weights) |
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

Secure logistic regression with a bias, trained by batch (`"bgd"`) or mini-batch (`"mbgd"`) gradient descent on the
cross-entropy. The activation is evaluated by Chebyshev interpolation or by Decor, through
[`sequre.stdlib.nonlinear`](../user-guide/nonlinear-functions.md). The default follows the variant: for `"binary"` the
shares' modulus, at each `fit`, `predict` and `loss` (`method_for(X)`): Decor on ring Sharetensors (11% faster than
Chebyshev there, and exact) where Decor admits the interval (its default interval always), Chebyshev on field
Sharetensors (about 3.4 times faster there, since Decor's sigmoid divides with the field's division), in the clear, on
the multiparty types (on their own arithmetic) and for an interval only Chebyshev admits, such as (−50, 10); a
`degree` makes it Chebyshev. `interval_for(X)` gives the interval a call uses; `interval` stays `(0, 0)` for this
default without an explicit interval. Train on secret Sharetensors (or entirely in the clear): a public Sharetensor
operand of the secure matrix product is not supported. Chebyshev for `"heuristic-multinomial"`, and Decor for `"multinomial"` (the softmax trains faster
with Decor on both moduli: 4% on the field, 2.1 times on the ring).

### Construction

```python
from sequre.stdlib.learn.log_reg import LogReg

model = LogReg(initial_weights)                                        # binary: Decor on the ring, Chebyshev on the field
model = LogReg(initial_weights, method="decor")                        # binary, Decor's sigmoid on (-10, 10)
model = LogReg(initial_weights, variant="multinomial")                 # softmax over the classes, Decor's exp
model = LogReg(initial_weights, variant="multinomial", fixed_shift_softmax=4.0)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `initial_weights` | `T` | — | (features + 1) × classes; the bias row last |
| `optimizer` | `str` | `"bgd"` | `"bgd"` or `"mbgd"` |
| `interval` | `tuple[float, float]` | `(0, 0)` | the activation's interval; `(0, 0)` picks the default below |
| `variant` | `str` | `"binary"` | `"binary"`, `"multinomial"` or `"heuristic-multinomial"` |
| `method` | `str` | `""` | `"chebyshev"` or `"decor"`; `""`: the variant's default (Decor for `"multinomial"`, Chebyshev for `"heuristic-multinomial"`, the shares' modulus for `"binary"`) |
| `degree` | `int` | `0` | Chebyshev degree (`0`: the default, 20) |
| `fixed_shift_softmax` | `float` | `-1.0` | `>= 0`: shift the softmax by this public offset instead of each row's maximum (no comparisons) |

### Variants

| Variant | Activation | Default interval: Chebyshev / Decor |
|---|---|---|
| `"binary"` | $\sigma(z)$, input clipped to the interval | (−50, 10) / (−10, 10) at F = 32 (for `method=""`, the one of the method each call uses) |
| `"multinomial"` | the softmax of each row of $\tilde{X}w$ | (−20, 0] / (−16, 0] |
| `"heuristic-multinomial"` | the shifted exp, not normalized (the earlier behavior, kept for comparison) | (−20, 0] / (−16, 0] |

The multinomial variants evaluate the exponent on $(-W, 0]$: the row maximum is subtracted first (`shift="max"`), or
the public `fixed_shift_softmax` when it is set (the rows' maxima must then stay near it; see `softmax` in
[Nonlinear Functions](../user-guide/nonlinear-functions.md)).

### Methods

| Method | Signature | Description |
|---|---|---|
| `fit` | `fit(mpc, X, y, step, epochs, verbose=False)` | Train the model. Returns `self`. |
| `predict` | `predict(mpc, X)` | Class probabilities: the activation of $\tilde{X}w$ |
| `loss` | `loss(mpc, X, y)` | The cross-entropy, with the bias column as in training; its log is accurate on (1e-4, 1] (`clipped_log`) |
| `method_for` | `method_for(X)` | The method a call on `X` uses (`"decor"` or `"chebyshev"`): the stored one; for the binary default, Decor on ring Sharetensors where Decor admits the interval, Chebyshev otherwise |
| `interval_for` | `interval_for(X)` | The activation's interval a call on `X` uses: the given one, or that method's default |
| `randomize_weights` | `randomize_weights(mpc, distribution="uniform")` | Reinitialize the weights randomly |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `coef_` | `T` | The learned weights, (features + 1) × classes (the bias row last) |
| `optimizer`, `variant`, `method`, `degree` | | As constructed |
| `interval` | `tuple[float, float]` | The activation's interval: the given one, or the variant's and method's default when `(0, 0)` is passed; the binary default (`method=""`) keeps `(0, 0)` and picks per call (`interval_for`) |
| `fixed_shift_softmax` | `float` | The softmax's public shift (`-1`: the row maximum) |

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
| `iterative(mpc, data, miss, data_mean, top_k, oversampling)` | Build a sketch from data with missing values via random bucketing (buckets and signs from the global stream: the same at every party) |
| `vectorized(mpc, data, sketch_size)` | Build a sketch using random matrix multiplication (the matrix from the global stream: the same at every party) |
| `generate_sketch_matrix(shape)` | A random sketching matrix: 0/1 entries, each row scaled to sum to 1 (a row of zeros is redrawn). It draws from the live stream: call it inside `mpc.randomness.global_stream()`, at every party (CP0 included), for a matrix every party agrees on, as `vectorized` and `random_pca_without_projection` do |

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
| `fit(mpc, data, labels, miss_rows, miss_col, ...)` | Run full MI pipeline: impute `factor` times, fit `FM` on each, combine via Rubin's rules. `dump_imputed_data` is a debugging aid that reveals every imputed dataset to every party |
| `MI.rubin(mpc, weights)` | Static: combine weight estimates using Rubin's combining rules |

On `MPU` data held in partitions (`"partition"` mode), each party's imputed values go to that party alone: one `reveal_local` of the predictions per imputation decrypts each party's rows to it, and each writes them into its own partition, in the clear. No other party learns them, and the traffic grows with the number of missing rows. The owner learns the imputation model's predictions for its rows (plus the imputation noise), which is what it imputes with.

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

### `compile`

```python
model = Sequential(layers).compile(mpc, loss="binary_crossentropy", optimizer="bgd", loss_interval=(1e-4, 1.0))
```

`loss_interval` is the interval of the cross-entropies' log: predictions are clipped into it, and its lower end must be
at least `1e-4`. A sigmoid output with `"binary_crossentropy"`, or a softmax output with `"categorical_crossentropy"`,
backpropagates the fused gradient $\hat{y} - y$ (exact, no division; with the softmax it needs label rows that sum to
1, one-hot or soft). Softmax has no element-wise derivative, so it is
accepted only as the output layer with the categorical cross-entropy. Nothing is revealed except, with `verbose`, the
training loss summed over the records (and over the output columns on Sharetensors; on the multiparty types one sum per
output column).

### Layers

_Defined in `stdlib/sequre/stdlib/learn/neural_net/layers.codon`_

#### `Input[ctype]`

The input layer. Holds no trainable parameters; it passes the data through.

```python
Input[type(X)](size)
```

#### `Dense[ctype]`

A fully connected layer computing $f(s \cdot (xW + b))$.

```python
Dense[type(X)](activation, size, kernel_initializer="uniform", bias_initializer="uniform",
               kernel_interval=(0.0, 1.0), kernel_scale=1.0, pre_act_scale=1.0, method="chebyshev",
               activation_interval=(0.0, 0.0), activation_degree=0, activation_clip=True,
               activation_reduction=False)
```

| Parameter | Default | Description |
|---|---|---|
| `activation` | — | one of the activations below |
| `size` | — | number of neurons |
| `kernel_initializer`, `bias_initializer` | `"uniform"` | `"uniform"`, `"normal"`, `"zeros"` or `"ones"` |
| `kernel_interval`, `kernel_scale` | `(0, 1)`, `1` | map the initializer's draw onto `kernel_interval`, then scale it (for example, SIREN's $U(-\sqrt{6/n}/\omega, \sqrt{6/n}/\omega)$); computed locally, no communication |
| `pre_act_scale` | `1` | the factor $s$ (SIREN's $\omega$); the backward pass carries it too |
| `method` | `"chebyshev"` | `"chebyshev"` (Sequre's method before Decor, as for binary `LogReg`) or `"decor"` (exact up to the fixed-point encodings; on the field its dividing activations cost a few times more, on the ring about the same) |
| `activation_interval` | `(0, 0)` | the activation's interval (`(0, 0)`: the method's default) |
| `activation_degree` | `0` | Chebyshev degree (`0`: the default) |
| `activation_clip` | `True` | clip pre-activations to the interval first (not for Decor's periodic sin, cos, tan, cot) |
| `activation_reduction` | `False` | Chebyshev sin and cos only: reduce the argument modulo $2\pi$ first, so the interpolant runs on $(-\pi, (2n_{cp} + 1)\pi)$ (replaces `activation_interval`; nothing is clipped). `Sharetensor` layers on ring shares and `ndarray` layers; other layer types raise `ValueError` at construction, field shares when the weights are drawn (at `compile`, before any round) |

A training forward pass keeps the activation's derivative for the backward pass; on Sharetensor and ndarray layers it
is the clipped function's beyond a clipped interval: 0 for the saturating activations, $0.5 (1 + t)$ for gelu and $s$ for
silu (their inner tanh or sigmoid is clipped). Other layer types keep $f'$ at the clipped point, and gelu's and silu's
inner term there (gelu's grows like $|x|^3$ beyond the inner interval).
On Sharetensor layers Decor's sin and cos with a `pre_act_scale` fold the scale into Decor's own scaling, and sin keeps
cos from the same opening. With the default method a sin, cos, tan or cot layer saturates beyond
(−π, π), (−π, π), (−1.2, 1.2) or (0.4, 2.7): size `activation_interval` to the pre-activations, or use
`method="decor"` (any argument). On the field a Decor sigmoid, tanh, tan, cot, silu or gelu layer costs a few times a
Chebyshev one (the field's division); on the ring about as much as the default Chebyshev (degree 20, which errs by
about $2 \times 10^{-3}$ for sigmoid and $4 \times 10^{-3}$ for tanh, against about $4 \times 10^{-6}$ for Decor), and
less than a Chebyshev interpolant as accurate as Decor.

Weight updates use **Nesterov accelerated gradient** with the `momentum` passed to `fit`.

### Activations

_Defined in `stdlib/sequre/stdlib/learn/neural_net/activations.codon`_

Each activation `f` comes as `f(mpc, x, method="chebyshev", interval, degree, clip)`, its derivative `df(...)`, and
`f_with_derivative(...)`, which returns both from one evaluation where it can. `activate`, `dactivate` and
`activate_with_derivative` dispatch on a runtime name. The neural-net stack defaults to Chebyshev (the nonlinear module
to Decor, and for sigmoid and tanh to `"auto"`).

| Activation | Function | Evaluated by |
|---|---|---|
| `"linear"` | $x$ | — |
| `"relu"`, `"leaky_relu"` | $\max(0, x)$; $x$ for $x > 0$, $0.01x$ otherwise | comparisons |
| `"hardtanh"`, `"relu6"` | $\min(\max(x, -1), 1)$; $\min(\max(x, 0), 6)$ | comparisons (one call for both masks) |
| `"asigmoid"` | $x / \sqrt{x^2 + 1}$ | square root and division |
| `"sigmoid"`, `"tanh"`, `"exp"` | | `method` (Decor or Chebyshev) |
| `"sin"`, `"cos"`, `"tan"`, `"cot"` | radians | `method`; Decor takes any $x$ |
| `"sinh"`, `"cosh"` | | `method` |
| `"gelu"` | $\frac{x}{2}\left(1 + \tanh\left(\sqrt{2/\pi}\,(x + 0.044715x^3)\right)\right)$ | composed, by `method` |
| `"silu"` | $x\,\sigma(x)$ | composed, by `method` |
| `"softmax"` | each row | Decor's division; its exp by `method`; output layer only, with the categorical cross-entropy |
| `"erf"` | | Chebyshev under either method; its derivative's exp by `method` |

### Losses

_Defined in `stdlib/sequre/stdlib/learn/neural_net/loss.codon`_

Each loss is averaged over the records.

| Loss | Function | Derivative |
|---|---|---|
| `"hinge"` | $\max(0, 1 - y\hat{y})$ | $-y \cdot \mathbf{1}[1 - y\hat{y} > 0]$ |
| `"mse"` | $(y - \hat{y})^2 / 2$ | $\hat{y} - y$ |
| `"binary_crossentropy"` | $-(y \log \hat{y} + (1 - y)\log(1 - \hat{y}))$ | $(\hat{y} - y) / (\hat{y}(1 - \hat{y}))$; $\hat{y} - y$ fused with a sigmoid output |
| `"categorical_crossentropy"` | $-\sum y \log \hat{y}$ | $-y / \hat{y}$; $\hat{y} - y$ fused with a softmax output |

The cross-entropies clip $\hat{y}$ into `loss_interval` and take an accurate log there (`clipped_log`: the Chebyshev
interpolant refined by Newton's iteration through Decor's exp).

### Examples

#### Credit score classification — [applications/credit_score.codon](https://github.com/0xTCG/sequre/blob/main/applications/credit_score.codon)

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

#### Drug-target interaction — [applications/dti.codon](https://github.com/0xTCG/sequre/blob/main/applications/dti.codon)

DTI inference with a larger network (128 neurons) over 8192-dimensional feature vectors, using mini-batch gradient descent.

```python
layers = (
    Input[type(X)](8192),
    Dense[type(X)]("relu", 128, "normal", "zeros"),
    Dense[type(X)]("linear", 1, "normal", "zeros"))

model = Sequential(layers).compile(mpc, loss="hinge", optimizer="bgd")
model.fit(mpc, X=X, y=y, step=0.2, epochs=50, momentum=0.9, batch_size=16)
```

#### SIREN — [applications/siren.codon](https://github.com/0xTCG/sequre/blob/main/applications/siren.codon)

An implicit neural representation with periodic activations: two sine layers and a linear output fit an image as a
function of its pixel coordinates. `siren_model` builds and compiles it for ndarray and Sharetensor (the tested
types; training on MPU layers is an open defect of `Sequential`), with SIREN's
initialization and $\omega$ as `pre_act_scale`.

```python
from applications.siren import synthetic_image, siren_model

X, y = synthetic_image(8)                     # coordinates in [-1, 1]^2, pixels in [-1, 1]
mX, my = Sharetensor.enc(mpc, X), Sharetensor.enc(mpc, y)
with mpc.randomness.global_stream():          # every party draws the same initial weights
    model = siren_model(mpc, type(mX), 2, 16, 30.0)             # Decor's sin
model.fit(mpc, X=mX, y=my, step=0.01, epochs=4, momentum=0.9)
```

The backward pass carries $\omega$ (the chain rule), so a step that trains the plaintext model trains the secure one:
step 0.01 with momentum 0.9 gets an 8 × 8 image below $10^{-4}$ in 100 epochs for every initialization tried, while
0.03 stalls for some. With `method="chebyshev"`, the interpolant needs enough degree for the interval's periods: on the
default interval (about (−90, 90) at $\omega = 30$) degree 20 is off by about 2. `degree=0` picks the smallest multiple
of 10 within `SIREN_SIN_TOLERANCE` ($10^{-6}$): 120 on (−90, 90). With `reduce_interval=True` (ring shares; the field
raises `ValueError` when the model is built) the argument is reduced modulo $2\pi$ first, and degree 30 suffices for up
to four compute parties.

#### GWAS by logistic association — [applications/gwas_logreg.codon](https://github.com/0xTCG/sequre/blob/main/applications/gwas_logreg.codon)

One binary logistic regression per SNP, on the SNP, the covariates and a bias. `gwas_logreg` returns each SNP's
coefficient (its log odds ratio) and the fraction of individuals its model classifies right; only those aggregates are
revealed.

```python
from applications.gwas_logreg import load_gwas_data, gwas_logreg

geno, pheno, cov = load_gwas_data("geno.txt", "pheno.txt", "cov.txt", standardize_covariates=True)
w0 = array([[0.0] for _ in range(cov.shape[1] + 2)])  # the SNP's, the covariates' and the bias' coefficients
betas, accuracies = gwas_logreg(mpc, Sharetensor.enc(mpc, geno), Sharetensor.enc(mpc, pheno),
                                Sharetensor.enc(mpc, cov), Sharetensor.enc(mpc, w0))  # Decor on the ring, Chebyshev on the field
```

Genotypes are allele counts (0, 1 or 2; a missing call raises `ValueError`). `step` (default 1.0) is per individual,
so one default suits any cohort size; it assumes features of unit scale, hence `standardize_covariates`. The defaults
train 50 epochs per SNP on the interval (−10, 10), where Decor's sigmoid is exact and Chebyshev's degree-20 interpolant
is within $1.8 \times 10^{-3}$. The method defaults to LogReg's binary default: Decor on ring shares (11% faster there
in the benchmarks), Chebyshev on field shares (about 3.5 times faster there); pass `method="decor"` or `"chebyshev"`
to fix it.
