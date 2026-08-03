# Neural networks: Sequre vs CrypTen vs MPyC

Frameworks present: Sequre (64b), Sequre (128b), Sequre (192b), CrypTen (64b, TFP), CrypTen (64b, TTP), MPyC (128b).

Two sequential feed-forward networks, trained for 2 epochs of batch gradient descent with Nesterov momentum, from initial weights shared across every framework; 3 timed repetitions per cell, median reported. `n` is the number of training rows -- for SIREN that is the pixel count of a square image, for the MLP the batch of a full-batch step.

Missing frameworks show `--`. `_skipped_` means the runner declined that cell; its reason is in the row's `note` in `results/`.

### Lines of model code

_non-blank, non-comment lines a user must write to define and train the network in that framework; library code is not counted. See `loc.py` for exactly what is inside the markers._

| model | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|
| siren | 21 | 21 | 21 | 29 | 29 | -- |
| mlp | 18 | 18 | 18 | 21 | 21 | 50 |

### Training time

_seconds per epoch, median of timed repetitions_

| model | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|---|
| siren | 64 | 0.0198 | 0.0312 | 0.0581 | 0.0555 | 0.0761 | _skipped_ |
| siren | 256 | 0.0620 | 0.0738 | 0.1424 | 0.0716 | 0.0964 | _skipped_ |
| siren | 1024 | 0.2324 | 0.2956 | 0.5710 | 0.2082 | 0.2315 | _skipped_ |
| siren | 4096 | 0.9076 | 1.2731 | 2.7152 | 0.8276 | 0.6299 | _skipped_ |
| mlp | 8 | 0.1650 | 0.3001 | 0.4457 | 0.0436 | 0.0542 | 153.1 |
| mlp | 32 | 0.4599 | 0.7620 | 1.0324 | 0.0563 | 0.0742 | _skipped_ |
| mlp | 128 | 1.7243 | 2.9440 | 3.7914 | 0.1191 | 0.1541 | _skipped_ |
| mlp | 512 | 10.2 | 26.7 | 27.1 | 0.4250 | 0.4779 | _skipped_ |

### Communication

_MiB sent per training run by the reporting party_

| model | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|---|
| siren | 64 | 2.5 | 5.0 | 7.5 | 11.5 | 11.5 | _skipped_ |
| siren | 256 | 8.3 | 16.4 | 24.5 | 45.3 | 45.3 | _skipped_ |
| siren | 1024 | 31.2 | 61.9 | 92.6 | 180.5 | 180.5 | _skipped_ |
| siren | 4096 | 123.1 | 244.0 | 364.9 | 721.4 | 721.4 | _skipped_ |
| mlp | 8 | 22.6 | 43.5 | 60.1 | 9.8 | 9.8 | 1,391.2 |
| mlp | 32 | 41.2 | 76.2 | 93.9 | 17.5 | 17.5 | _skipped_ |
| mlp | 128 | 115.5 | 206.8 | 229.2 | 48.5 | 48.5 | _skipped_ |
| mlp | 512 | 412.9 | 729.6 | 770.2 | 172.5 | 172.5 | _skipped_ |

### Accuracy (max absolute prediction error)

_trained network's predictions vs the float64 reference's_

| model | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|---|
| siren | 64 | 3.1e+01 | 2.0e-08 | 2.7e-08 | 1.2e-02 | 1.2e-02 | _skipped_ |
| siren | 256 | 1.3e+00 | 3.4e-08 | 3.6e-08 | 1.3e-02 | 1.3e-02 | _skipped_ |
| siren | 1024 | 1.3e+02 | 7.9e-08 | 6.0e-08 | 1.3e-02 | 1.3e-02 | _skipped_ |
| siren | 4096 | 1.9e+01 | 1.0e-07 | 1.2e-07 | 1.5e-02 | 1.4e-02 | _skipped_ |
| mlp | 8 | 2.7e-04 | 3.9e-09 | 5.1e-09 | 1.9e-04 | 2.2e-04 | 3.7e-09 |
| mlp | 32 | 2.8e-04 | 4.3e-09 | 4.6e-09 | 2.4e-04 | 2.7e-04 | _skipped_ |
| mlp | 128 | 2.7e-04 | 5.7e-09 | 4.5e-09 | 3.0e-04 | 3.0e-04 | _skipped_ |
| mlp | 512 | 3.7e-04 | 5.0e-09 | 6.6e-09 | 3.2e-04 | 2.7e-04 | _skipped_ |

### Accuracy (final training loss)

_compare against the float64 reference below_

| model | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|---|
| siren | 64 | 219.434875 | 0.028500 | 0.028500 | 0.028351 | 0.028366 | _skipped_ |
| siren | 256 | 0.387863 | 0.032983 | 0.032983 | 0.032974 | 0.032990 | _skipped_ |
| siren | 1024 | 4235.916150 | 0.034618 | 0.034618 | 0.034592 | 0.034637 | _skipped_ |
| siren | 4096 | -32011.247200 | 0.035359 | 0.035359 | 0.035355 | 0.035385 | _skipped_ |
| mlp | 8 | 0.425034 | 0.424964 | 0.424964 | 0.425003 | 0.425018 | 0.424964 |
| mlp | 32 | 0.471909 | 0.471300 | 0.471300 | 0.471283 | 0.471344 | _skipped_ |
| mlp | 128 | 0.490814 | 0.485618 | 0.485618 | 0.485641 | 0.485626 | _skipped_ |
| mlp | 512 | 0.531143 | 0.491912 | 0.491912 | 0.491913 | 0.491913 | _skipped_ |

### Sequre (192b) speedup

_median wall-clock of the baseline divided by Sequre (192b)'s; &gt;1 means Sequre is faster. Note the differing share widths in the column labels._

| model | n | vs CrypTen (64b, TFP) | vs CrypTen (64b, TTP) | vs MPyC (128b) |
|---|---|---|---|---|
| siren | 64 | 0.955x | 1.3x | -- |
| siren | 256 | 0.503x | 0.677x | -- |
| siren | 1024 | 0.365x | 0.405x | -- |
| siren | 4096 | 0.305x | 0.232x | -- |
| mlp | 8 | 0.0978x | 0.122x | 343.6x |
| mlp | 32 | 0.0546x | 0.0719x | -- |
| mlp | 128 | 0.0314x | 0.0407x | -- |
| mlp | 512 | 0.0157x | 0.0176x | -- |

## Online cost (dealer detached)

_Sequre rows have the dealer detached via `mpc.detach_dealer()`, so only the compute parties' work is timed. CrypTen and MPyC have no separable offline phase and repeat their end-to-end numbers here. Large cells are skipped: a detached dealer buffers every byte it would have sent until the block exits, and a full training run exceeds what the transport sustains._

### Training time, dealer detached

_seconds per epoch, median of timed repetitions_

| model | n | Sequre (64b) | Sequre (128b) | Sequre (192b) | CrypTen (64b, TFP) | CrypTen (64b, TTP) | MPyC (128b) |
|---|---|---|---|---|---|---|---|
| siren | 64 | 0.0226 | 0.0378 | 0.0739 | 0.0555 | 0.0761 | _skipped_ |
| siren | 256 | 0.0628 | 0.1412 | 0.2172 | 0.0716 | 0.0964 | _skipped_ |
| siren | 1024 | 0.2223 | 0.3381 | 0.5324 | 0.2082 | 0.2315 | _skipped_ |
| siren | 4096 | _skipped_ | _skipped_ | _skipped_ | 0.8276 | 0.6299 | _skipped_ |
| mlp | 8 | 0.1505 | 0.7255 | 0.7691 | 0.0436 | 0.0542 | 153.1 |
| mlp | 32 | 0.4006 | 1.1660 | 1.5149 | 0.0563 | 0.0742 | _skipped_ |
| mlp | 128 | _skipped_ | _skipped_ | _skipped_ | 0.1191 | 0.1541 | _skipped_ |
| mlp | 512 | _skipped_ | _skipped_ | _skipped_ | 0.4250 | 0.4779 | _skipped_ |

### Sequre (192b) speedup (dealer detached)

_median wall-clock of the baseline divided by Sequre (192b)'s; &gt;1 means Sequre is faster. Note the differing share widths in the column labels._ Sequre is detached; the baselines are not.

| model | n | vs CrypTen (64b, TFP) | vs CrypTen (64b, TTP) | vs MPyC (128b) |
|---|---|---|---|---|
| siren | 64 | 0.751x | 1.0x | -- |
| siren | 256 | 0.33x | 0.444x | -- |
| siren | 1024 | 0.391x | 0.435x | -- |
| siren | 4096 | -- | -- | -- |
| mlp | 8 | 0.0567x | 0.0704x | 199.1x |
| mlp | 32 | 0.0372x | 0.049x | -- |
| mlp | 128 | -- | -- | -- |
| mlp | 512 | -- | -- | -- |

### float64 reference

_`ref.py` trains the same networks in the clear, from the same initial weights. Every accuracy column above is measured against this._

| model | n | final loss |
|---|---|---|
| siren | 64 | 0.028500 |
| siren | 256 | 0.032983 |
| siren | 1024 | 0.034618 |
| siren | 4096 | 0.035359 |
| mlp | 8 | 0.424964 |
| mlp | 32 | 0.471300 |
| mlp | 128 | 0.485618 |
| mlp | 512 | 0.491912 |

### Run metadata

- `crypten-ttp_CP0.jsonl`: optimizer=bgd-nesterov, framework=crypten, parties=2, provider=TTP, protocol=beaver, precision_bits=16
- `crypten-ttp_CP1.jsonl`: optimizer=bgd-nesterov, framework=crypten, parties=2, provider=TTP, protocol=beaver, precision_bits=16
- `crypten_CP0.jsonl`: optimizer=bgd-nesterov, framework=crypten, parties=2, provider=TFP, protocol=beaver, precision_bits=16
- `crypten_CP1.jsonl`: optimizer=bgd-nesterov, framework=crypten, parties=2, provider=TFP, protocol=beaver, precision_bits=16
- `mpyc.jsonl`: optimizer=bgd-nesterov, framework=mpyc, parties=3, threshold=1, sectype=SecFxp(64), note=network built from MPyC primitives; MPyC ships no neural-network layer
- `sequre_w128_CP0.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w128_CP1.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w128_CP2.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=128, nbit_k=64, nbit_f=32, nbit_v=28, modulus=ring_2^127
- `sequre_w192_CP0.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w192_CP1.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w192_CP2.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=192, nbit_k=64, nbit_f=32, nbit_v=64, modulus=ring_2^191
- `sequre_w64_CP0.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
- `sequre_w64_CP1.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
- `sequre_w64_CP2.jsonl`: optimizer=bgd-nesterov, framework=sequre, parties=3, mpc_int_size=64, nbit_k=32, nbit_f=16, nbit_v=10, modulus=ring_2^63
