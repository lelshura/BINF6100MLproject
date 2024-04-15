import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load the original dataset
filename = '/Users/lubainakothari/Documents/BINF6100MLproject/rynazal_abundance_metadata_tranformation.csv'
df = pd.read_csv(filename, header=0)

# Specify categorical and numeric features
categorical_features = ['gender', 'country']
numeric_features = df.drop(['Sample ID', 'CRC', 'gender', 'country'], axis=1).columns.tolist()

# Create preprocessors for numeric and categorical data
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')

# Create a column transformer to apply the transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply transformations
X_processed = preprocessor.fit_transform(df.drop(['Sample ID', 'CRC'], axis=1))
y = df['CRC'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=0.80, random_state=25, stratify=y)

# Define the AdaBoost model with default base estimator
model = AdaBoostClassifier(algorithm='SAMME', random_state=25)

# Define the grid of values to search
grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

# Define the evaluation procedure
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

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
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

# Print the evaluation results
print("Test Set Accuracy: %.3f" % accuracy)
print("Confusion Matrix:\n", conf_mat)
print("ROC AUC Score: %.3f" % roc_auc)
print("Classification Report:\n", classification_report(y_test, y_pred))

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
axes[0].grid()
axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
axes[0].legend(loc="best")

# Scalability plot
axes[1].grid()
axes[1].plot(train_sizes, fit_times_mean, 'o-')
axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                     fit_times_mean + fit_times_std, alpha=0.1)
axes[1].set_xlabel("Training examples")
axes[1].set_ylabel("fit_times")
axes[1].set_title("Scalability of the model")

# Performance plot
axes[2].grid()
axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1)
axes[2].set_xlabel("fit_times")
axes[2].set_ylabel("Score")
axes[2].set_title("Performance of the model")

plt.show()