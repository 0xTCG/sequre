# Privacy-Preserving Sepsis Prediction using Simple RNN

## 1. Project Overview
This project builds a secure Machine Learning pipeline using the **Sequre framework**. By using Multi-Party Computation (MPC), the code ensures complete data privacy while the model runs. While the specific test case focuses on predicting Sepsis, the main goal of this addition is to build and test Recurrent Neural Network (RNN) features inside Sequre's encrypted environment.

## 2. Dataset & Preprocessing
The model uses the **MIMIC-III** clinical database for testing. 
* **Data Extraction:** Patient records were filtered into Sepsis and Non-Sepsis groups.
* **Feature Engineering:** The data was formatted into sequential time-steps to represent a 48-hour patient observation window.
* **Weight Generation:** Model weights were trained, extracted, and saved as CSV matrices (`Wx.csv`, `Wh.csv`, `b.csv`, `Wy.csv`, `by.csv`) so they could be loaded into Sequre.

## 3. Model Architecture: Simple RNN
A standard Simple RNN was built using the `@sequre` decorator, which allows math operations on securely shared tensors.
* **Sequence Length:** 48 time-steps.
* **Hidden State:** Starts as a 1x64 zeros tensor and updates sequentially.
* **Recurrence Logic:** At each time step `t`, the hidden state updates using the input data, the memory from the previous step, and the bias.

## 4. Sequre Framework Implementation Details
To manage the limits of fixed-point math inside the framework, specific Sequre standard library functions were added:
* **Data Encryption:** Input data and model weights are encrypted across the compute parties (CP0, CP1, CP2) right at initialization.
* **Overflow Management:** The `clip` function from `sequre.stdlib.builtin` is applied to the raw hidden state tensors. This stops the fixed-point numbers from overflowing and crashing during the 48-step loop.
* **Activation Function:** The `chebyshev_sigmoid` function is used as a secure replacement for the standard sigmoid activation. 

## 5. Conclusion
This code shows how to integrate sequential models (RNNs) into the Sequre framework. It creates a baseline for processing sensitive, time-series data (like health records) while keeping strict cryptographic privacy between all computing parties.