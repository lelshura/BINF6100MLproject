# ========================================================================================= #
# Trust Your Gut: Classifying Colorectal Cancer Diagnosis with Microbiome Genomic Profiling #
# ========================================================================================= #
# Authors:   Neha Patel, Lubaina Kothari, Lina Elshurafa
# Acknowledgements: Dr. Dan Tulpan
# Date:     April 18th, 2024

# How to run:   python3  master_script.py  data.csv
# ========================================================================================= #

from sklearn.model_selection import learning_curve, train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, roc_curve, auc, classification_report)
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Set global Matplotlib style parameters
plt.rcParams.update({'font.size': 12})
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)

#-------------------------------------------------------------------------------------------------------------------------------------------------
#                                         Define Functions
#-------------------------------------------------------------------------------------------------------------------------------------------------
def load_data(filename):
    """ Load dataset from file """
    df = pd.read_csv(filename, header=0)
    data = df.values
    X = data[:, 1:-1] # Sample ID and CRC removed
    y = data[:, -1].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

    print("X_train, X_test size: ", X_train.shape, X_test.shape)
    print("y_train, y_test size: ", y_train.shape, y_test.shape)
    print("\nX_train:\n", X_train, "\n")
    print("X_test:\n", X_test, "\n")
    print("y_train:\n", y_train, "\n")
    print("y_test:\n", y_test, "\n")
    return X_train, X_test, y_train, y_test, df

def train_and_evaluate(model_search, X_train, y_train, X_test):
    """ Train the model and evaluate it on the test set """
    model_search.fit(X_train, y_train)
    best_model = model_search.best_estimator_
    predictions = best_model.predict(X_test)
    probabilities = best_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class

    # Print the best parameters and the best cross-validation score
    print("Best parameters:", model_search.best_params_)
    print("Best cross-validation score: {:.2f}".format(model_search.best_score_))

    return predictions, probabilities, best_model


def plot_learning_curve(estimator, X, y, title="Learning Curves Logistic Regression", axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), filename=None):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    if filename:
        plt.savefig(filename)
    plt.show()

def calculate_and_print_metrics(y_test, y_pred, y_probs, model_name):
    """ Calculate and print metrics """
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_probs)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Specificity calculation from confusion matrix
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)

    # Output the metrics
    print(f"Metrics for {model_name}:")
    print("Confusion Matrix:\n", conf_matrix)
    print("F1 Score:", f1)
    print("Matthews Correlation Coefficient:", mcc)
    print("ROC AUC:", roc_auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Specificity:", specificity)

def plot_roc_curve(y_test, y_probs, model_name):
    """ Plot ROC curves """
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'ROC_curve_{model_name}.png')
    plt.show()

#----------------------------------------------------------------------------------------------------------------------------------------
#                                         Main Program
#----------------------------------------------------------------------------------------------------------------------------------------
def main(args):
    # Extract filename from args
    filename = args.filename
    # Load data and set cross-validation
    X_train, X_test, y_train, y_test, df = load_data(filename)
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    #-------------------------------------|SVM Classifier|-------------------------------------------------------------------------------
    # Define the model
    svm_model = svm.SVC(probability = True)

    # Define the parameters to search
    svm_param = {
        'C': np.logspace(-4, 4, 20),  # Log-uniform distribution
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'gamma': np.logspace(-4, 4, 20),  # Applies to non-linear kernels
        'degree': [2, 3, 4, 5]  # Applies to 'poly' kernel
    }

    # Create a RandomizedSearchCV object
    svm_search = RandomizedSearchCV(svm_model, svm_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

    # Train and evaluate model
    svm_predictions, svm_probabilities, svm_best_model = train_and_evaluate(svm_search, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, svm_predictions, svm_probabilities, 'SVM')
    plot_roc_curve(y_test, svm_probabilities, 'SVM')
    plot_learning_curve(svm_best_model, X_train, y_train, title="Learning Curve for SVM", filename='svm_learning_curve.png')

    #-------------------------------------|Logistic Regression Classifier|---------------------------------------------------------------
    # Define model
    logreg_model = LogisticRegression(max_iter=100000)

    # Define the parameters to search
    logreg_param = [
        {'C': np.logspace(-4, 4, 20), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
        {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs', 'sag', 'saga']},
        {'C': np.logspace(-4, 4, 20), 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': np.linspace(0, 1, 10)},
        {'solver': ['lbfgs', 'sag', 'saga'], 'penalty': [None]}
    ]

    # Create a RandomizedSearchCV object
    logreg_search = RandomizedSearchCV(logreg_model, logreg_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

    # Train and evaluate model
    logreg_predictions, logreg_probabilities, logreg_best_model = train_and_evaluate(logreg_search, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, logreg_predictions, logreg_probabilities, 'Logistic Regression')
    plot_roc_curve(y_test, logreg_probabilities, 'Logistic Regression')
    plot_learning_curve(logreg_best_model, X_train, y_train, title="Learning Curve for Logistic Regression", filename='logreg_learning_curve.png')

    #-------------------------------------|Random Forest Classifier|---------------------------------------------------------------------
    # Define the model
    rf_model = RandomForestClassifier(random_state=25)

    # Define the parameters to search
    rf_param = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=600, num=10)],
                   'max_features': ['sqrt', 'log2', 0.5, None],
                   'max_depth': [int(x) for x in np.linspace(10, 60, num=11)],
                   'min_samples_split': [2, 4, 6, 8, 10],
                   'min_samples_leaf': [1, 2, 3, 4, 5],
                   'bootstrap': [True, False]}


    # Create a RandomizedSearchCV object
    rf_search = RandomizedSearchCV(rf_model, rf_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

    # Train and evaluate model
    rf_predictions, rf_probabilities, rf_best_model = train_and_evaluate(rf_search, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, rf_predictions, rf_probabilities, 'Random Forest')
    plot_roc_curve(y_test, rf_probabilities, 'Random Forest')
    plot_learning_curve(rf_best_model, X_train, y_train, title="Learning Curve for Random Forest", filename='rf_learning_curve.png')

    #-------------------------------------|AdaBoost Classifier|--------------------------------------------------------------------------

    # Define the AdaBoost model with default base estimator
    adab_model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME', random_state=25)

    # Define the parameters to search
    adab_param = {
        'n_estimators': [10, 50, 100, 500],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
    }

    # Create a GridSearch object
    adab_grid = GridSearchCV(estimator=adab_model, param_grid=adab_param, n_jobs=-1, cv=cv, scoring='roc_auc')

    # Train and evaluate model
    adab_predictions, adab_probabilities, adab_best_model = train_and_evaluate(adab_grid, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, adab_predictions, adab_probabilities, 'AdaBoost')
    plot_roc_curve(y_test, adab_probabilities, 'AdaBoost')
    plot_learning_curve(adab_best_model, X_train, y_train, title="Learning Curve for AdaBoost", filename='adaboost_learning_curve.png')

    # Feature Importance
    feature_importances = adab_best_model.feature_importances_
    # Create a pandas series with feature importances and labels, then sort it
    importances = pd.Series(feature_importances, index=df.columns[1:-1])
    sorted_features = importances.sort_values(ascending=False)
    # Select the top 10 features
    top_importances = sorted_features[:10]

    # Create the plot with the specified aesthetics
    plt.figure(figsize=(10, 6))
    top_importances.plot(kind='barh', color='skyblue')

    # Invert y-axis to have the highest importance at the top
    plt.gca().invert_yaxis()

    plt.title('Top 10 Feature Importances in AdaBoost Model')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')

    # Tight layout to improve the spacing between subplots
    plt.tight_layout()

    plt.show()
    #-------------------------------------|MLP Classifier|-------------------------------------------------------------------------------
    # Standardize the features
    scaler = StandardScaler()
    x_train_mlp = scaler.fit_transform(X_train)
    x_test_mlp = scaler.transform(X_test)

    # Define the model
    mlp_model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                          random_state=25, max_iter=1000, alpha=0.0001,
                          solver='adam', verbose=10, n_iter_no_change=10,
                          early_stopping=True, validation_fraction=0.1)

    # Train and evaluate model
    mlp_predictions, mlp_probabilities, mlp_best_model = train_and_evaluate(x_train_mlp, y_train, x_test_mlp)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, mlp_predictions, mlp_probabilities, 'MLP')
    plot_roc_curve(y_test, mlp_probabilities, 'MLP')
    plot_learning_curve(mlp_best_model, x_train_mlp, y_train, title="Learning Curve for MLP",
                        filename='mlp_learning_curve.png')

#----------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a machine learning model on provided dataset.")
    parser.add_argument('filename', type=str, help="Path to the dataset file.")
    args = parser.parse_args()

    main(args.filename)
