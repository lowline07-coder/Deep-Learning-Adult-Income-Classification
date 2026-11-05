# Deep Learning – Adult Income Classification Using ANN

## Project Overview

This project focuses on building and optimizing an Artificial Neural Network (ANN) using Deep Learning techniques to predict whether an individual's annual income exceeds $50K based on demographic and employment-related attributes.

The project uses the Adult Income Dataset and explores different neural network configurations through systematic experiments. It investigates how activation functions, weight initialization techniques, loss functions, batch normalization, and optimizers influence model convergence and classification performance.

The primary objective is to understand the impact of different Deep Learning techniques and identify an effective ANN configuration for binary classification.

## Objectives

- Perform data cleaning, preprocessing, and exploratory data analysis.
- Build a baseline Artificial Neural Network using TensorFlow and Keras.
- Compare different activation functions and understand their impact on learning.
- Experiment with weight initialization techniques.
- Evaluate different loss functions for binary classification.
- Handle class imbalance using weighted Binary Cross-Entropy and Focal Loss.
- Investigate the impact of Batch Normalization.
- Compare optimizers based on convergence speed and performance.
- Develop a final optimized ANN model.
- Evaluate classification performance using appropriate metrics.

## Dataset

**Dataset:** Adult Income Dataset (Census Income Dataset)

The dataset contains demographic, educational, and employment-related information used to predict an individual's income category.

**Target Variable:** `income`

| Class | Description |
|---|---|
| 0 | Income ≤50K |
| 1 | Income >50K |

### Important Features

- Age
- Workclass
- Education
- Education Number
- Marital Status
- Occupation
- Relationship
- Race
- Sex
- Capital Gain
- Capital Loss
- Hours per Week
- Native Country

## Technologies & Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- TensorFlow
- Keras
- Jupyter Notebook

## Project Workflow

### 1. Data Loading, Cleaning & Exploratory Data Analysis

- Load and inspect the Adult Income Dataset.
- Examine dataset dimensions, data types, and statistical information.
- Remove unnecessary whitespace from categorical features.
- Handle missing values by replacing `?` with NaN.
- Remove rows containing missing values.
- Encode the target variable into binary classes.
- Perform exploratory data analysis using visualizations.

### 2. Data Preprocessing

- Apply One-Hot Encoding to categorical features.
- Use `drop_first=True` to avoid redundant dummy variables.
- Apply StandardScaler to numerical features:
  - Age
  - Education Number
  - Capital Gain
  - Capital Loss
  - Hours per Week
- Perform an 80:20 train-test split.
- Use stratified sampling to preserve class distribution.

### 3. Baseline Artificial Neural Network

Build a baseline ANN using the following architecture:

- Input Layer: Based on the number of processed features.
- Hidden Layer 1: 128 neurons.
- Hidden Layer 2: 64 neurons.
- Output Layer: 1 neuron with Sigmoid activation.
- Optimizer: Adam.
- Loss Function: Binary Cross-Entropy.

The baseline model provides a reference point for subsequent experiments.

### 4. Activation Function Comparison

Train and compare four different activation functions:

- ReLU
- Tanh
- Sigmoid
- ELU

Analyze their training and validation accuracy, convergence behaviour, and class-1 F1-score.

Additional investigations include:

- Dead Neuron Analysis using ReLU.
- Gradient Flow Analysis using Sigmoid.
- Understanding the Vanishing Gradient Problem.

### 5. Weight Initialization Techniques

Compare five weight initialization methods:

- Glorot Uniform
- Glorot Normal
- He Uniform
- He Normal
- Zeros Initialization

Investigate their influence on:

- Training convergence.
- Weight distributions.
- Gradient stability.
- Model performance.

A separate experiment demonstrates the symmetry problem associated with zero weight initialization.

### 6. Loss Function Experiments

Explore different loss functions for binary classification:

**Binary Cross-Entropy (BCE)**

Used as the primary loss function for binary classification.

**Mean Squared Error (MSE)**

Compared against BCE to understand why regression-oriented loss functions may be less suitable for binary classification.

**Weighted Binary Cross-Entropy**

Uses class weights to address the imbalance between income categories.

**Focal Loss**

Focuses training on difficult and misclassified examples by reducing the contribution of easy examples.

Compare their impact on minority-class Precision, Recall, and F1-score.

### 7. Batch Normalization

Investigate the effect of Batch Normalization on ANN training.

Experiments include:

- Comparing baseline ANN with Batch Normalization.
- Comparing Batch Normalization before and after activation.
- Inspecting learned Gamma and Beta parameters.
- Analyzing training and validation convergence.

The two layer arrangements investigated are:

- Dense → BatchNormalization → ReLU
- Dense → ReLU → BatchNormalization

### 8. Optimizer Comparison

Compare five optimization configurations:

- Stochastic Gradient Descent (SGD)
- SGD with Momentum
- RMSprop
- Adam
- Adam with Explicit Hyperparameters

Investigate their convergence behaviour and learning rate sensitivity.

A separate experiment compares SGD and Adam using different learning rates.

### 9. Final Combined ANN Model

Combine the selected techniques into a final ANN architecture.

**Final Model Configuration:**

| Component | Configuration |
|---|---|
| Hidden Layers | 128 → 64 |
| Activation Function | ReLU |
| Weight Initialization | He Normal |
| Batch Normalization | Enabled |
| Optimizer | Adam |
| Loss Function | Binary Cross-Entropy |
| Class Weighting | Conditional |
| Training Epochs | 80 |
| Batch Size | 64 |
| Output Activation | Sigmoid |

Class weighting is selected only when Weighted BCE improves the minority-class F1-score compared with the baseline.

## Model Evaluation

The models are evaluated using the following metrics:

- **Accuracy:** Measures the overall proportion of correct predictions.
- **Precision:** Measures how many predicted positive cases are actually positive.
- **Recall:** Measures how many actual positive cases are correctly identified.
- **F1-Score:** Provides a balance between Precision and Recall.
- **ROC-AUC:** Measures the model's ability to distinguish between the two income classes.

Additional evaluation techniques include:

- Confusion Matrix
- Classification Report
- Training and Validation Loss Curves
- Training and Validation Accuracy Curves
- ROC Curve Comparison

Special attention is given to the minority class (`>50K`) because of the class imbalance present in the dataset.

## Project Structure

```text
Deep-Learning-Adult-Income-Classification/
│
├── DL_Pr2.ipynb
│
├── adult.csv.zip
│
└── README.md
```

**Note:** The dataset file should be placed in the appropriate location before executing the notebook. Update the dataset path if necessary.

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/Deep-Learning-Adult-Income-Classification.git
```

### 2. Navigate to the Project Directory

```bash
cd Deep-Learning-Adult-Income-Classification
```

### 3. Install Required Libraries

```bash
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow jupyter
```

### 4. Launch Jupyter Notebook

```bash
jupyter notebook
```

### 5. Run the Notebook

Open `DL_Pr2.ipynb` and execute the cells sequentially.

Make sure the dataset path is correctly configured before running the notebook.

## Key Learnings

Through this project, the following Deep Learning concepts are explored:

- Artificial Neural Networks and Multi-Layer Perceptrons.
- Forward Propagation and Backpropagation.
- Activation Functions and their mathematical behaviour.
- Vanishing Gradients and Dead Neurons.
- Weight Initialization and the Symmetry Problem.
- Binary Cross-Entropy and Focal Loss.
- Class Imbalance Handling.
- Batch Normalization.
- Optimizer Selection and Convergence.
- Learning Rate Sensitivity.
- Model Evaluation and Performance Comparison.

## Conclusion

This project provides a practical understanding of how different Deep Learning techniques influence the training and performance of Artificial Neural Networks.

By systematically experimenting with activation functions, initialization methods, loss functions, normalization techniques, and optimizers, the project demonstrates the importance of selecting appropriate configurations for binary classification problems.

The final combined ANN brings together the selected configurations and evaluates their effectiveness using multiple classification metrics, with particular emphasis on minority-class performance.

---

**Project Domain:** Deep Learning | Artificial Neural Networks | Binary Classification

**Tools:** Python | TensorFlow | Keras | Scikit-learn | Pandas | NumPy
