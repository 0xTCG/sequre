"""float64 reference training loop, in numpy.

This is the correctness witness every runner is measured against: the same
architecture, the same data, the same initial weights and the same update rule,
computed in the clear. A secure framework that is fast and wrong shows up here
as a large `max_abs_pred_err` in the same row as its timing.

The update rule is transcribed from Sequre's `Sequential`
(`stdlib/sequre/stdlib/learn/neural_net/model.codon`) rather than taken from a
textbook, because the point of comparison is Sequre's semantics:

  * loss and its derivative are pre-normalised by the number of rows,
  * the gradient step is Nesterov accelerated, in the two-line `v`-then-`W`
    form that Sequre uses,
  * a layer computes `f(s * (xW + b))` and the public factor `s` reappears in
    the backward pass.

The other runners reproduce these too, so an accuracy gap between frameworks is
a protocol or precision result and never an optimizer difference.
"""

from __future__ import annotations

import numpy as np

import spec


def activate(name: str, x: np.ndarray) -> np.ndarray:
    if name == "sin":
        return np.sin(x)
    if name == "relu":
        return x * (x > 0)
    if name == "linear":
        return x
    raise ValueError(f"unknown activation: {name}")


def dactivate(name: str, x: np.ndarray) -> np.ndarray:
    if name == "sin":
        return np.cos(x)
    if name == "relu":
        return (x > 0).astype(np.float64)
    if name == "linear":
        return np.ones_like(x)
    raise ValueError(f"unknown activation: {name}")


class Reference:
    """Sequre's Sequential, in float64."""

    def __init__(self, model: str):
        self.spec = spec.MODELS[model]
        initial = spec.initial_weights(model)
        self.w = [np.array(w, dtype=np.float64) for w, _ in initial]
        self.b = [np.array(b, dtype=np.float64) for _, b in initial]
        self.vw = [np.zeros_like(w) for w in self.w]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.pre = [p for _, _, p, _ in self.spec["layers"]]
        self.act = [a for a, _, _, _ in self.spec["layers"]]

    def forward(self, X: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Returns the pre-activations and the outputs of every layer."""
        pre_acts, outputs = [], []
        out = X
        for i in range(len(self.w)):
            z = out @ self.w[i] + self.b[i]
            if self.pre[i] != 1.0:
                z = z * self.pre[i]
            out = activate(self.act[i], z)
            pre_acts.append(z)
            outputs.append(out)
        return pre_acts, outputs

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)[1][-1]

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        # Matches Sequre's `loss(...).reveal(mpc).sum()`: elementwise
        # (y - out)^2 / 2, pre-normalised by the row count, then summed.
        out = self.predict(X)
        return float((((y - out) ** 2) / 2 / len(y)).sum())

    def step(self, X: np.ndarray, y: np.ndarray, step: float, momentum: float) -> None:
        pre_acts, outputs = self.forward(X)
        n_layers = len(self.w)

        dhidden = (outputs[-1] - y) / len(y)
        dw: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]
        db: list[np.ndarray] = [None] * n_layers  # type: ignore[list-item]

        for i in range(n_layers - 1, -1, -1):
            dhidden = dhidden * dactivate(self.act[i], pre_acts[i])
            if self.pre[i] != 1.0:
                dhidden = dhidden * self.pre[i]
            prev_out = X if i == 0 else outputs[i - 1]
            dw[i] = prev_out.T @ dhidden
            db[i] = dhidden.sum(axis=0, keepdims=True)
            if i > 0:
                dhidden = dhidden @ self.w[i].T

        for i in range(n_layers):
            vw_prev = self.vw[i].copy()
            self.vw[i] = self.vw[i] * momentum - dw[i] * step
            self.w[i] = self.w[i] + self.vw[i] * (momentum + 1) - vw_prev * momentum

            vb_prev = self.vb[i].copy()
            self.vb[i] = self.vb[i] * momentum - db[i] * step
            self.b[i] = self.b[i] + self.vb[i] * (momentum + 1) - vb_prev * momentum

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int, step: float, momentum: float) -> None:
        for _ in range(epochs):
            self.step(X, y, step, momentum)


def train(model: str, n: int) -> tuple[np.ndarray, float, np.ndarray]:
    """(predictions, final loss, target) after spec.EPOCHS of reference training."""
    X_raw, y_raw = spec.data(model, n)
    X = np.array(X_raw, dtype=np.float64)
    y = np.array(y_raw, dtype=np.float64)

    net = Reference(model)
    net.fit(X, y, spec.EPOCHS, spec.MODELS[model]["step"], spec.MOMENTUM)
    return net.predict(X), net.loss(X, y), y
