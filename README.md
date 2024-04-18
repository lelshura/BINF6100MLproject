# Trust Your Gut: Colorectal Cancer Diagnosis with Microbiome Genomic Profiling
Date: April 18th, 2024

## About the Project
### Part A:
In this project, we aim to use machine learning models to identify and classify individuals with colorectal cancer based on bacterial relative abundances in their gut microbiome.
The classification models trained are Support Vector Machine (SVM), Logistic Regression, Random Forest, AdaBoost, and Multilayer Perceptron (MLP).

### Part B:
In the second part of the project, an exploratory analysis is conducted by incorporating patient metadata features, including country, age, sex, and BMI, to analyze the effects of the additional features on model performance. The top-performing classification models from Part A are trained, Random Forest, AdaBoost, and SVM.

## Function Descriptions
### Part A
#### 1. *def load_data(filename)*
#### **For Data Loading and Preprocessing:** Loads a dataset from a CSV file and splits the dataset into training and testing sets.

#### 2. *def train_and_evaluate(model_search, X_train, y_train, X_test)*
**For Model Training:** Trains a machine learning model using the parameters defined in 'model_search' and evaluates it on the test set.

#### 3. *def plot_learning_curve(estimator, X, y, title="Learning Curves Logistic Regression", axes=None, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename=None)*
**For Plotting:** Generates learning curves to visually assess model performance and understand how the model performance improves with the addition of more training data.

#### 4. *def calculate_and_print_metrics(y_test, y_pred, y_probs, model_name)*
**For Calculating Metrics for Evaluation:** Calculate several performance metrics to evaluate models based on a Confusion Matrix, F1 Score, Matthews Correlation Coefficient, ROC AUC, accuracy, precision, recall, and specificity.

#### 5. *def plot_roc_curve(y_test, y_probs, model_name)*
**For Plotting:** Generates Receiver Operating Characteristic (ROC) curves to visually assess model performance at various thresholds.

#### 6. *def main(args)*
**For Program Execution:** Sets cross-validation and hyperparameter tuning and runs the previously defined functions on each model. Feature importance is also calculated for the top-performing models, Random Forest and AdaBoost. A SHAP analysis is also conducted for Random Forest to further study feature importance.

### Part B
In the second part of our analysis, functions 2-6, as defined for Part A, are used. 

## Getting Started
### Requirements
Python 3 and pip must be installed on your system to use this script.
This script also requires the following Python packages:
NumPy
Pandas
Scikit-Learn
Matplotlib
Shap

### Installation
Download and install Python from the official website (installing Python from this site will typically include pip). 
https://www.python.org/downloads/

Install the required Python libraries using pip:
pip install numpy pandas scikit-learn matplotlib shap

### Executing program
How to run:   python3  master_script.py  data.csv

## Authors
Neha Patel:
npatel31@uoguelph.ca

Lubaina Kothari:
lkothari@uoguelph.ca

Lina Elshurafa:
lelshura@uoguelph.ca

## Acknowledgements
Special thanks to Dr. Dan Tulpan for guidance and insights throughout the development of this project.
