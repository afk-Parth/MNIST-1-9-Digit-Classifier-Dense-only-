# MNIST 1‑9 Digit Classifier (Dense‑only) 

Fully‑connected neural network (no CNN) that classifies hand‑written digits **1‑9** from grayscale MNIST images.

| Item                | Details |
|---------------------|---------|
| **Dataset**         | MNIST – filtered to digits 1‑9 (60 000 train / 10 000 test → 54 000 / 9 000 after filtering) |
| **Input**           | 28×28 pixels → flattened to 784 |
| **Architecture**    | Dense(256 ReLU) → Dropout(0.3) → Dense(128 ReLU) → Dropout(0.3) → Dense(9 Softmax) |
| **Optimizer**       | Adam |
| **Accuracy**        | ⬜ 0.96 |

---

## Highlights
* **Pure MLP** – demonstrates that even without convolutions, a dense network can classify digits reasonably well.  
* **Data Filtering** – removed digit 0 to create a 9‑class problem (1‑9).  
* **Visualization** – notebook shows sample inputs and highlights first 10 mis‑classifications in red.

---

## How to Run
```bash
pip install tensorflow matplotlib scikit-learn
python mnist_dense_1to9.ipynb   # or open in Google Colab
