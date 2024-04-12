import sklearn
print(sklearn.__version__)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Define the location of the dataset
filename = '/Users/lubainakothari/Documents/BINF6100MLproject/rynazal_filtered_abundance.csv'

# Load the dataset; header is the first row
df = pd.read_csv(filename, header=0)

# Exclude 'Sample ID' and 'CRC' from the features, 'CRC' is the target variable
X = df.drop(['Sample ID', 'CRC'], axis=1).values
y = df['CRC'].values

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

print("X_train, X_test size: ", X_train.shape, X_test.shape)
print("y_train, y_test size: ", y_train.shape, y_test.shape)
print("\nX_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")
print("y_train:\n", y_train, "\n")
print("y_test:\n", y_test, "\n")

# Define the AdaBoost model with default base estimator
model = AdaBoostClassifier(algorithm='SAMME', random_state=25)

# Define hyperparameters to search
param_grid = {
    'n_estimators': [10, 50, 100, 500],
    'learning_rate': [0.001, 0.01, 0.1, 1.0]
}

# Set up the grid search with Stratified K-Fold cross-validation
cv = StratifiedKFold(n_splits=10, random_state=25, shuffle=True)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='accuracy')

# Execute the grid search on the training data
grid_result = grid.fit(X_train, y_train)

# Report the best configuration
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Define model with the best parameters
best_model = grid_result.best_estimator_

# Evaluate the model on the test set
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])

# Print the evaluation results
print("Test Set Accuracy: %.3f" % accuracy)
print("Confusion Matrix:\n", conf_mat)
print("ROC AUC Score: %.3f" % roc_auc)
print("Classification Report:\n", classification_report(y_test, y_pred))
