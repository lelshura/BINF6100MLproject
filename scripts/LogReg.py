from sklearn.model_selection import learning_curve, train_test_split, RandomizedSearchCV, GridSearchCV
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, f1_score, matthews_corrcoef,
                             roc_auc_score, accuracy_score, precision_score,
                             recall_score, roc_curve, auc)

# Define functions
def plot_learning_curve(estimator, X, y, title="Learning Curves SVM", axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 4))

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


#Main Program ----
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

# Define model
model = LogisticRegression(max_iter=100000)

# Parameter grid for RandomizedSearchCV
param_distributions_lr = [
    {'C': np.logspace(-4, 4, 20), 'penalty': ['l1', 'l2'], 'solver': ['liblinear']},
    {'C': np.logspace(-4, 4, 20), 'penalty': ['l2'], 'solver': ['lbfgs', 'sag', 'saga']},
    {'C': np.logspace(-4, 4, 20), 'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': np.linspace(0, 1, 10)},
    {'solver': ['lbfgs', 'sag', 'saga'], 'penalty': [None]}
]

# Create a RandomizedSearchCV object
random_search_lr = RandomizedSearchCV(model, param_distributions_lr, n_iter=100, cv=5, scoring='roc_auc', random_state=42, verbose=2)

# Fit RandomizedSearchCV to the data
random_search_lr.fit(X_train, y_train)

# Results of search 
print("Best parameters:", random_search_lr.best_params_)
print("Best cross-validation score: {:.2f}".format(random_search_lr.best_score_))

# Predict using the test set
best_model_lr = random_search_lr.best_estimator_
y_pred = best_model_lr.predict(X_test)
y_probs = best_model_lr.predict_proba(X_test)[:, 1]

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
print("Confusion Matrix:\n", conf_matrix)
print("F1 Score:", f1)
print("Matthews Correlation Coefficient:", mcc)
print("ROC AUC:", roc_auc)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("Specificity:", specificity)

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], 'r--')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/ROC_curveLR.png')
plt.show()

# Plot learning curves
plot_learning_curve(model, X_train, y_train, n_jobs=-1, ylim=(0.5, 1.01))
plt.savefig('../figures/Learning_curvesLR.png')
plt.show()
