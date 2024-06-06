# Evaluate a random forest algorithm for classification
# Reference: https://machinelearningmastery.com/random-forest-ensemble-in-python/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve, roc_auc_score, auc
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import shap
import time

# Define functions
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(10, 3))

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


"""
Train and evaluate initial random forest model
"""

# Define dataset
filename = '../filtered_data/rynazal_filtered_abundance.csv'

# Load the dataset; header is first row
df = pd.read_csv(filename, header=0)

# Extract input (X) and output (y) variables
data = df.values
X = data[:, 1:-1]
y = data[:, -1].astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

# Define the model
model = RandomForestClassifier(random_state=25)

# Train and evaluate the model with cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=25)
n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')

# Report performance based on mean and standard deviation of accuracy scores from cross validation
print('Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# Based on best model from cross validation, train the model again on entire training dataset
model.fit(X_train, y_train)

# Predict output values based on the trained model with input values from the test set
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:,1]


#predictions_custom = (y_probs >= 0.6).astype(int)


# Determine accuracy scores of predictions of the test set
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.3f}')

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_matrix)

# Display a classification report
print(classification_report(y_test, y_pred))

# Determine AUC
roc_auc = roc_auc_score(y_test, y_pred)
print("ROC AUC for og model:", roc_auc)

'''
Hyperparameter Tuning
'''
# Show default parameters
print('Parameters currently in use:\n')
pprint(model.get_params())

# define hyperparameter search space
n_estimators = [int(x) for x in np.linspace(start=100, stop=600, num=10)]
max_features = ['sqrt', 'log2', 0.5, None]
max_depth = [int(x) for x in np.linspace(10, 60, num=11)]
max_depth.append(None)
min_samples_split = [2, 4, 6, 8, 10]
min_samples_leaf = [1, 2, 3, 4, 5]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)

start_time = time.time()

model_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=5, verbose=1,
                                  random_state=25, n_jobs=-1)

model_random.fit(X_train, y_train)

end_time = time.time()
duration = end_time - start_time
print(f"Hyperparameter tuning took {duration:.2f} seconds.")

best_model = model_random.best_estimator_
print("Best params of tuned model", model_random.best_params_)

# Evaluation
y_pred_tuned = best_model.predict(X_test)
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
conf_matrix_tuned = confusion_matrix(y_test, y_pred_tuned)


f1_tuned = f1_score(y_test, y_pred_tuned)
mcc_tuned = matthews_corrcoef(y_test, y_pred_tuned)
precision_tuned = precision_score(y_test, y_pred_tuned)
recall_tuned = recall_score(y_test, y_pred_tuned)
y_probs_tuned = best_model.predict_proba(X_test)[:,1]
fpr_tuned, tpr_tuned, thresholds = roc_curve(y_test, y_probs_tuned)
roc_auc_tuned = auc(fpr_tuned, tpr_tuned)

# Specificity calculation from confusion matrix
tn_tuned, fp_tuned, fn_tuned, tp_tuned = conf_matrix_tuned.ravel()
specificity_tuned = tn_tuned / (tn_tuned + fp_tuned)

# Output the metrics
print("Confusion Matrix:\n", conf_matrix_tuned)
print("F1 Score:", f1_tuned)
print("Matthews Correlation Coefficient:", mcc_tuned)
print("ROC AUC:", roc_auc_tuned)
print("Accuracy:", accuracy_tuned)
print("Precision:", precision_tuned)
print("Recall:", recall_tuned)
print("Specificity:", specificity_tuned)

# Plot ROC curve

fpr_tuned, tpr_tuned, thresholds = roc_curve(y_test, y_probs_tuned)
roc_auc_tuned = auc(fpr_tuned, tpr_tuned)
print("ROC AUC for tuned model:", roc_auc_tuned)

plt.figure(figsize=(8, 6))  # Optionally specify the figure size
plt.plot(fpr_tuned, tpr_tuned, lw=2, label=f'AUC = {roc_auc_tuned:.2f}')  # Adjust color to blue
plt.plot([0, 1], [0, 1], 'r--', label='Chance (area = 0.50)')  # Red dashed line for the chance line
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/roc_curve_tuned_ab.png')  # Save the figure
plt.close()

# Plot Feature Importance
importances = best_model.feature_importances_
feature_importances = pd.Series(importances, index=df.columns[1:-1])
sorted_features = feature_importances.sort_values(ascending=False)
top_ten_features = sorted_features[:10]

plt.figure(figsize=(8, 6))  # Set the figure size as needed

# Sort the features in ascending order for display
sorted_features = top_ten_features.sort_values(ascending=True)
sorted_features.plot(kind='barh', color='skyblue')

plt.title('Top 10 Feature Importances Using Mean Decrease Gini')
plt.ylabel('Features')
plt.xlabel('Mean Decrease Gini')
plt.tight_layout()
plt.savefig('../figures/feature_importance_ab.png', bbox_inches='tight')
plt.close()


# Plot learning curve

# Assuming the previous code blocks have been executed and `best_model` is available
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Call the function
learning_curve_fig = plot_learning_curve(estimator=best_model, title='Learning Curves (Random Forest)',
                    X=X_train, y=y_train, axes=axes, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=-1)

learning_curve_fig.savefig('../figures/learning_curve.png')
plt.close()

# SHAP analysis

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

class_index = 1  # or 0, depending on which class you are interested in
# This selects the SHAP values for the chosen class across all features and samples
class_specific_shap_values = shap_values[:, :, class_index]

# Calculate mean absolute SHAP values across all samples for the chosen class
mean_abs_shap_values = np.abs(class_specific_shap_values).mean(axis=0)

# Identify top 10 feature indices
sorted_feature_indices = np.argsort(mean_abs_shap_values)[::-1][:10]

# Extract SHAP values for top 10 features
top_shap_values = class_specific_shap_values[:, sorted_feature_indices]

# Corresponding feature names for these top 10 features
top_feature_names = df.columns[1:-1].to_numpy()[sorted_feature_indices]

# Generate a summary plot for the top 10 features for the selected class
shap.summary_plot(top_shap_values, X_test[:, sorted_feature_indices], feature_names=top_feature_names.tolist())
plt.savefig('../figures/shap_summary_plot_top_10_class_' + str(class_index) + '.png')
plt.close()