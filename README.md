# 🌳 Decision Tree Classifier from Scratch

![Language](https://img.shields.io/badge/Python-3.8%2B-blue)
![Libraries](https://img.shields.io/badge/Libraries-NumPy%20%7C%20Pandas-green)
![Topic](https://img.shields.io/badge/Algorithm-ID3%20%2F%20CART-orange)

## 📌 Project Overview
This repository contains a purely Pythonic implementation of a **Decision Tree Classifier** built from scratch without relying on high-level ML libraries for the core logic. 

The project demonstrates a deep understanding of tree-based algorithms by manually implementing:
* **Splitting Criteria:** Entropy (Information Gain) and Gini Impurity.
* **Tree Construction:** Recursive partitioning for both categorical and numerical features.
* **Prediction Logic:** Traversing the learned tree structure to classify new samples.

It also includes a detailed **Manual Tracing Report** comparing the custom implementation against `sklearn.tree.DecisionTreeClassifier`.

## ⚙️ Core Features
* **Custom Split Logic:** Finds the optimal split by maximizing Information Gain or minimizing Gini Impurity.
* **Support for Mixed Data:** Handles both continuous (numerical) and categorical features automatically.
* **Configurable Hyperparameters:**
    * `max_depth`: Limits tree growth to prevent overfitting.
    * `min_samples_split`: Controls the minimum size of a node to attempt a split.
    * `min_information_gain`: Threshold for valid splits.
* **Performance Metrics:** Includes a custom confusion matrix evaluation function.

## 🧮 Mathematical Foundations
The implementation is based on the following concepts (detailed in `docs/Manual_Calculation_Report.pdf`):

### 1. Entropy
$$E(S) = \sum_{i=1}^{c} -p_i \log_2 p_i$$

### 2. Gini Impurity
$$Gini = 1 - \sum_{i=1}^{c} (p_i)^2$$

### 3. Information Gain
$$Gain(S, A) = Entropy(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Entropy(S_v)$$

## 🚀 How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/mariamashraf731/Decision-Tree-From-Scratch.git](https://github.com/mariamashraf731/Decision-Tree-From-Scratch.git)
    ```
2.  **Install requirements:**
    ```bash
    pip install pandas numpy scikit-learn
    ```
3.  **Run the script:**
    ```bash
    python src/decision_tree.py
    ```
## 👨‍💻 Technologies Used
* **Python**: Core logic.
* **NumPy & Pandas**: efficient data manipulation.
* **Scikit-Learn**: Used only for benchmarking and confusion matrix calculation.