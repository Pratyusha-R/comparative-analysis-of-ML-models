# Comparing Different Machine Learning Models on Cervical Cancer Dataset

This project compares the performance of various machine learning algorithms to classify the risk of cervical cancer based on relevant features.

## Project Overview

This project aims to identify the most effective model for predicting cervical cancer risk. It explores various machine learning algorithms, including ensemble methods and neural networks, to assess their performance on a healthcare-related dataset.

## Dataset

The **Cervical Cancer Dataset** was used for this analysis. This dataset includes features relevant to cervical cancer risk prediction, allowing a detailed examination of model behavior across different algorithmic approaches.
Dataset used: https://archive.ics.uci.edu/dataset/383/cervical+cancer+risk+factors

## Algorithms Covered

The following models were implemented and evaluated:

1. **Random Forest Classifier**  
   - Ensemble method using bootstrap aggregating and random feature selection.
   
2. **Support Vector Machine (SVM)**  
   - Uses a hyperplane to maximize margin separation with optional kernel functions for non-linear separations.
   
3. **Logistic Regression**  
   - Models the probability of class membership using a logistic function and maximum likelihood estimation.
   
4. **k-Nearest Neighbors (k-NN)**  
   - Instance-based learning algorithm that classifies based on the majority class of nearest neighbors.
   
5. **Gradient Boosting**  
   - Sequentially builds an ensemble of models to reduce errors via gradient descent.
   
6. **Neural Networks (Multi-Layer Perceptron - MLP)**  
   - Composed of layers of neurons with forward and backpropagation for optimization.
   
7. **Decision Trees**  
   - Recursive partitioning based on impurity criteria to generate leaf node predictions.
   
8. **AdaBoost (Adaptive Boosting)**  
   - Boosting algorithm that adjusts weights on data instances and weak learners iteratively.
   
9. **Bagging Classifier**  
   - Ensemble method that trains multiple models on bootstrap samples for aggregated predictions.

## Project Structure

- **Data Preparation:** Steps to load, clean, and preprocess the data.
- **Model Implementation:** Code to implement each machine learning model with brief explanations.
- **Evaluation:** Comparative analysis of model performance with metrics like accuracy, precision, recall, and F1-score.

## Requirements

- Python 3.x
- Required libraries: `scikit-learn`, `pandas`, `numpy`, `matplotlib`

Install dependencies using:
```bash
pip install -r requirements.txt
```

## Usage

To run the project, execute the following:

```bash
# Clone the repository
git clone https://github.com/yourusername/Comparing-ML-Models-Cervical-Cancer.git

# Change directory
cd Comparing-ML-Models-Cervical-Cancer

# Run the notebook or script
# For Jupyter Notebook
jupyter notebook Comparing_different_models.ipynb
```
