from sklearn.model_selection import learning_curve, train_test_split, RandomizedSearchCV, GridSearchCV, RepeatedStratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, roc_curve, auc, classification_report)

#--------------------------------------------------------------------------------------------------------------------------------------------------------
#                                         Define Functions
#--------------------------------------------------------------------------------------------------------------------------------------------------------
def plot_learning_curve(estimator, X, y, title="Learning Curves Logistic Regression", axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
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

    return plt


def calculate_and_print_metrics(y_test, y_pred, y_probs, model_name):
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
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.title(f'ROC Curve for {model_name}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f'../figures/ROC_curve_{model_name}.png')
    plt.show()

def plot_model_learning_curve(model, X_train, y_train, title):
    plot_learning_curve(model, title, X_train, y_train, n_jobs=-1, ylim=(0.5, 1.01))
    plt.savefig(f'../figures/Learning_curves_{title}.png')
    plt.show()

#--------------------------------------------------------------------------------------------------------------------------------------------------------
#                                         Main Program
#--------------------------------------------------------------------------------------------------------------------------------------------------------
# Define the location of the dataset
# In practice use argparse to read the file as input parameter
filename='../filtered_data/rynazal_filtered_abundance.csv'

# Load the dataset; header is first row
df = pd.read_csv(filename, header=0)
data = df.values

# Exclude "Sample ID" and response variable ("CRC") columns from features
X = data[:,1:-1]
y = data[:,-1].astype(int)

# Split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

print("X_train, X_test size: ", X_train.shape, X_test.shape)
print("y_train, y_test size: ", y_train.shape, y_test.shape)
print("\nX_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")
print("y_train:\n", y_train, "\n")
print("y_test:\n", y_test, "\n")


# Define the evaluation procedure for cross-validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=1)

#-------------------------------------|SVM Classifier|-----------------------------------------------------------------------------------
# Define the model
svm_model = svm.SVC(probability = True)

# Setup the parameter grid to sample from during fitting
svm_param = {
    'C': np.logspace(-4, 4, 20),  # Log-uniform distribution
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
    'gamma': np.logspace(-4, 4, 20),  # Applies to non-linear kernels
    'degree': [2, 3, 4, 5]  # Applies to 'poly' kernel
}

# Create a RandomizedSearchCV object
svm_search = RandomizedSearchCV(svm_model, svm_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

# Fit RandomizedSearchCV to the data
svm_search.fit(X_train, y_train)

# Results of search
print("Best SVM parameters:", svm_search.best_params_)
print("Best SVM cross-validation score: {:.2f}".format(svm_search.best_score_))

# Predict using the test set
svm_best_model = svm_search.best_estimator_
svm_predictions = svm_best_model.predict(X_test)
svm_prob = svm_ best_model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class



#-------------------------------------|Logistic Regression Classifier|--------------------------------------------------------------------
# Define model
logreg_model = LogisticRegression(max_iter=100000)

# Parameter grid for RandomizedSearchCV
logreg_param = [
    {'C': np.logspace(-4, 4, 20), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
    {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs', 'sag', 'saga']},
    {'C': np.logspace(-4, 4, 20), 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': np.linspace(0, 1, 10)},
    {'solver': ['lbfgs', 'sag', 'saga'], 'penalty': [None]}
]

# Create a RandomizedSearchCV object
logreg_search = RandomizedSearchCV(logreg_model, logreg_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

# Fit RandomizedSearchCV to the data
logreg_search.fit(X_train, y_train)

# Results of search
print("Best parameters:", logreg_search.best_params_)
print("Best cross-validation score: {:.2f}".format(logreg_search.best_score_))

# Predict using the test set
logreg_best_model = logreg_search.best_estimator_
logreg_predictions = logreg_best_model.predict(X_test)
logreg_prob = logreg_best_model.predict_proba(X_test)[:, 1]


#-------------------------------------|Random Forest Classifier|--------------------------------------------------------------------------
# Define the model
rf_model = RandomForestClassifier(random_state=25)

# Train and evaluate the model with cross validation
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Report performance based on mean and standard deviation of accuracy scores from cross validation
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

rf_param = {'n_estimators': [int(x) for x in np.linspace(start=100, stop=600, num=10)],
               'max_features': ['sqrt', 'log2', 0.5, None],
               'max_depth': [int(x) for x in np.linspace(10, 60, num=11)],
               'min_samples_split': [2, 4, 6, 8, 10],
               'min_samples_leaf': [1, 2, 3, 4, 5],
               'bootstrap': [True, False]}

rf_search = RandomizedSearchCV(rf_model, rf_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

# Based on best model from cross validation, train the model again on entire training dataset
rf_search.fit(X_train, y_train)

# Predict output values based on the trained model with input values from the test set
rf_best_model = rf_search.best_estimator_
rf_predictions = rf_best_model.predict(X_test)
rf_prob = rf_best_model.predict_proba(X_test)[:, 1]


#-------------------------------------|AdaBoost Classifier|-------------------------------------------------------------------------------

# Define the AdaBoost model with default base estimator
adab_model = AdaBoostClassifier(algorithm='SAMME', random_state=25)

# Define the grid of values to search
adab_param = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

# Define the grid search procedure
adab_search = RandomizedSearchCV(adab_model, adab_param, n_jobs=-1, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2)

# Execute the grid search
adab_search.fit(X_train, y_train)

# Define best model from grid search
adab_best_model = adab_search.best_estimator_

# Evaluate the model on the test set
adab_predictions = adab_best_model.predict(X_test)
adab_prob = adab_best_model.predict_proba(X_test)[:, 1]

#-------------------------------------|MLP Classifier|------------------------------------------------------------------------------------



#-------------------------------------|Calculate Metrics and Plot|------------------------------------------------------------------------
# SVM
calculate_and_print_metrics(y_test, svm_predictions, svm_prob, 'SVM')
plot_roc_curve(y_test, svm_prob, 'SVM')
plot_model_learning_curve(svm_best_model, X_train, y_train, 'SVM')

# Logistic Regression
calculate_and_print_metrics(y_test, logreg_predictions, logreg_prob, 'Logistic Regression')
plot_roc_curve(y_test, logreg_prob, 'Logistic Regression')
plot_model_learning_curve(logreg_best_model, X_train, y_train, 'Logistic Regression')

# Random Forest
calculate_and_print_metrics(y_test, rf_predictions, rf_prob, 'Random Forest')
plot_roc_curve(y_test, rf_prob, 'Random Forest')
plot_model_learning_curve(rf_best_model, X_train, y_train, 'Random Forest')

# AdaBoost
calculate_and_print_metrics(y_test, adab_predictions, adab_prob, 'AdaBoost')
plot_roc_curve(y_test, adab_prob, 'AdaBoost')
plot_model_learning_curve(adab_best_model, X_train, y_train, 'AdaBoost')

#MLP
