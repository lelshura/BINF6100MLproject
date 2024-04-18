import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, matthews_corrcoef, \
    roc_curve
from sklearn.tree import DecisionTreeClassifier

# Update matplotlib settings for plotting
plt.rcParams.update({
    'font.size': 12,      # global font size
    'font.family': 'sans-serif',
    'axes.labelsize': 12,  # font size of the axes labels
    'axes.titlesize': 16,  # font size of the axes title
    'xtick.labelsize': 12,  # font size of the tick labels
    'ytick.labelsize': 12   # font size of the tick labels
})

# Define the location of the dataset
filename = '/Users/lubainakothari/Documents/BINF6100MLproject/rynazal_filtered_abundance.csv'

# Load the dataset; header is the first row
df = pd.read_csv(filename, header=0)

# Exclude 'Sample ID' and 'CRC' from the features, 'CRC' is the target variable
X = df.drop(['Sample ID', 'CRC'], axis=1).values
y = df['CRC'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

# Define the AdaBoost model with a Decision Tree as base estimator
model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME', random_state=25)

# Define the grid of values to search
grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

# Define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=25)

# Define the grid search procedure
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy')

# Execute the grid search
grid_result = grid_search.fit(X_train, y_train)

# Summarize the best score and configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Define best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
y_pred_proba = best_model.predict_proba(X_test)[:, 1]  # for ROC AUC
mcc = matthews_corrcoef(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the evaluation results
print("Test Set Accuracy: %.3f" % accuracy)
print("Matthews Correlation Coefficient:", mcc)
print("Confusion Matrix:\n", conf_mat)
print("ROC AUC Score: %.3f" % roc_auc)
print("Classification Report:\n", classification_report(y_test, y_pred))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='AdaBoost (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()

# Obtain the learning curve data
train_sizes, train_scores, test_scores, fit_times, score_times = learning_curve(
    best_model, X_train, y_train, cv=cv, n_jobs=-1,
    train_sizes=np.linspace(.1, 1.0, 5), return_times=True)

# Calculate the mean and standard deviation of the train and test scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Mean and std of scores
train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)
fit_times_mean = np.mean(fit_times, axis=1)
fit_times_std = np.std(fit_times, axis=1)

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Learning curve plot
axes[0].set_title("Learning Curve (AdaBoost)")
axes[0].set_xlabel("Training examples")
axes[0].set_ylabel("Score")
axes[0].set_ylim(0.5, 1.01)  # Set the limits for the Y-axis
axes[0].grid(True)
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
axes[0].legend(loc="best")

# Scalability plot
axes[1].grid(True)
axes[1].plot(train_sizes, fit_times_mean, 'o-')
axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                     fit_times_mean + fit_times_std, alpha=0.1)
axes[1].set_xlabel("Training examples")
axes[1].set_ylabel("fit_times")
axes[1].set_title("Scalability of the model")
axes[1].set_ylim(0)  # Start y-axis at 0 for the scalability plot

# Performance plot
axes[2].grid(True)
axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
axes[2].set_xlabel("fit_times")
axes[2].set_ylabel("Score")
axes[2].set_title("Performance of the model")

plt.show()

# Feature Importance
feature_importances = best_model.feature_importances_
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