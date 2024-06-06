from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint
import shap
import time
from sklearn.pipeline import Pipeline

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

def get_feature_names_out(column_transformer):
    """Get feature names from all transformers."""
    new_feature_names = []

    # Loop over all transformers in the Column Transformer
    for name, estimator, features in column_transformer.transformers_:
        if name == 'remainder':
            # The remainder are those columns not specified in transformers; they are passed as is
            new_feature_names.extend(column_transformer.feature_names_in_[features])
        elif hasattr(estimator, 'get_feature_names_out'):
            # If the estimator has 'get_feature_names_out', use it
            names = estimator.get_feature_names_out(features)
            new_feature_names.extend(names)
        else:
            # Otherwise, use the provided feature names directly
            new_feature_names.extend(features)

    return new_feature_names



# Load the original dataset
filename = '../filtered_data/rynazal_abundance_metadata.csv'
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
    ],
    remainder='passthrough'  # This will pass through any other columns not listed explicitly
)

# Apply transformations
X_processed = preprocessor.fit_transform(df.drop(['Sample ID', 'CRC'], axis=1))
y = df['CRC'].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, train_size=0.80, random_state=25, stratify=y)


# Define the model
model = RandomForestClassifier(random_state=25)

# Train and evaluate the model with cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=25)

'''
#Hyperparameter Tuning
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

print(random_grid)


model_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=100, cv=cv, verbose=1,
                                  random_state=25, n_jobs=-1)

model_random.fit(X_train, y_train)


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

plt.figure(figsize=(8, 6))  # Optionally specify the figure size
plt.plot(fpr_tuned, tpr_tuned, lw=2, label=f'AUC = {roc_auc_tuned:.2f}')  # Adjust color to blue
plt.plot([0, 1], [0, 1], 'r--', label='Chance (area = 0.50)')  # Red dashed line for the chance line
plt.title('ROC Curve for Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('../figures/roc_curve_tuned_metadata.png')  # Save the figure
plt.close()

# Plot Feature Importance
importances = best_model.feature_importances_
# Getting new feature names after transformations
new_feature_names = get_feature_names_out(preprocessor)

# Assuming `best_model` has been defined and trained
feature_importances = pd.Series(best_model.feature_importances_, index=new_feature_names)

sorted_features = feature_importances.sort_values(ascending=False)
top_ten_features = sorted_features[:10]

plt.figure(figsize=(12, 6))  # Increase figure size
top_ten_features.plot(kind='bar', color='skyblue')
plt.title('Top 10 Feature Importances Using Mean Decrease Gini')
plt.xlabel('Features')
plt.ylabel('Mean Decrease Gini')
plt.xticks(rotation=45)  # Rotate x-ticks to prevent overlapping
plt.tight_layout()  # Automatically adjust subplot params to give specified padding
plt.savefig('../figures/feature_importance.png', bbox_inches='tight')
plt.close()

# Plot learning curve

# Assuming the previous code blocks have been executed and `best_model` is available
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define cross-validation strategy
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=25)

# Call the function
learning_curve_fig = plot_learning_curve(estimator=best_model, title='Learning Curves (Random Forest)',
                    X=X_train, y=y_train, axes=axes, ylim=(0.7, 1.01),
                    cv=cv, n_jobs=-1)

learning_curve_fig.savefig('../figures/learning_curve.png')
plt.close()

# SHAP analysis

explainer = shap.TreeExplainer(best_model)
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
