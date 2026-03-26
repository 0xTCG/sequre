# API Reference

This section provides detailed reference documentation for every public type and module in Sequre.

## Core types

| Module | Description |
|---|---|
| [Sharetensor](sharetensor.md) | Additively secret-shared tensor — the core MPC data type |
| [Ciphertensor](ciphertensor.md) | Local CKKS ciphertext tensor with operator overloading |

## Multiparty types

| Module | Description |
|---|---|
| [Multiparty Types (overview)](multiparty-types.md) | Overview and comparison of MPU, MPP, MPA |
| [MPU](mpu.md) | Multiparty Union — highest-level distributed type (union of MPP and MPA) |
| [MPP](mpp.md) | Multiparty Partition — horizontally partitioned data across parties |
| [MPA](mpa.md) | Multiparty Aggregate — additive shares with optional encryption |

## Runtime & environment

| Module | Description |
|---|---|
| [MPC Instance](mpc-instance.md) | The runtime environment that orchestrates all protocols |
| [Decorators & Attributes](decorators.md) | `@sequre`, `@local`, `@online`, `@main`, `@flatten`, compiler IR pass attributes |

## Libraries

| Module | Description |
|---|---|
| [Secure Stdlib](stdlib.md) | Built-in secure functions: sign, abs, inv, sqrt, Chebyshev approximations, linear algebra, bit protocols |
| [Secure ML](learn.md) | Machine learning: linear/logistic regression, SVM, PCA, multiple imputation, neural networks |

## Infrastructure

| Module | Description |
|---|---|
| [Lattiseq (CKKS)](lattiseq.md) | Low-level homomorphic encryption engine |
| [Configuration](configuration.md) | Environment variables, constants, and compile-time settings |
