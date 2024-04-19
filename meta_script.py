# ========================================================================================= #
# Trust Your Gut: Classifying Colorectal Cancer Diagnosis with Microbiome Genomic Profiling #
# PART B: Incorporating Metadata                                                            #
# ========================================================================================= #
# Authors:   Neha Patel, Lubaina Kothari, Lina Elshurafa
# Acknowledgements: Dr. Dan Tulpan
# Date:     April 18th, 2024

# How to run:   python3  meta_script.py  data.csv
# ========================================================================================= #
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, GridSearchCV, learning_curve
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, matthews_corrcoef, precision_score, recall_score, roc_curve, roc_auc_score, auc, RocCurveDisplay
from sklearn.model_selection import learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
from pprint import pprint
import shap
import argparse
from sklearn.pipeline import Pipeline

# Set global Matplotlib style parameters
plt.rcParams.update({'font.size': 12})
plt.rc('font', size=12)
plt.rc('axes', titlesize=16)
plt.rc('xtick', labelsize=12)
plt.rc('ytick', labelsize=12)
#-------------------------------------------------------------------------------------------------------------------------------------------------
#                                         Define Functions
#-------------------------------------------------------------------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------------------------------------------------------
#                                         Main Program
#----------------------------------------------------------------------------------------------------------------------------------------

# Load the original dataset
filename = 'rynazal_abundance_metadata.csv'
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

# Train and evaluate the model with cross validation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=25)


def main(args):
    """
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
    rf_search = RandomizedSearchCV(rf_model, rf_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2, n_jobs=-1)

    # Train and evaluate model
    rf_predictions, rf_probabilities, rf_best_model = train_and_evaluate(rf_search, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, rf_predictions, rf_probabilities, 'Random Forest with Metadata')
    plot_roc_curve(y_test, rf_probabilities, 'Random Forest with Metadata')
    plot_learning_curve(rf_best_model, X_train, y_train, title="Learning Curve for Random Forest with Metadata", filename='rf_learning_curve_meta.png')

    # Plot Feature Importance
    rf_importance = rf_best_model.feature_importances_
    new_feature_names = get_feature_names_out(preprocessor)
    rf_feature_importances = pd.Series(rf_best_model.feature_importances_, index=new_feature_names)
    rf_sorted_features = rf_feature_importances.sort_values(ascending=False)
    rf_top_ten_features = rf_sorted_features[:10]

    # Sort the features in ascending order for display
    rf_sorted_features = rf_top_ten_features.sort_values(ascending=True)
    rf_sorted_features.plot(kind='barh', color='skyblue')

    plt.title('Top 10 Feature Importance Using Mean Decrease Gini')
    plt.ylabel('Features')
    plt.xlabel('Mean Decrease Gini')
    plt.xticks(rotation=45)  # Rotate x-ticks to prevent overlapping
    plt.tight_layout()
    plt.savefig('feature_importance_ab.png', bbox_inches='tight')
    plt.close()

    # SHAP analysis
    explainer = shap.TreeExplainer(rf_best_model)
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
    plt.savefig('shap_summary_plot_top_10_class_' + str(class_index) + '.png')
    plt.close()
    """

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
    calculate_and_print_metrics(y_test, adab_predictions, adab_probabilities, 'AdaBoost with Metadata')
    plot_roc_curve(y_test, adab_probabilities, 'AdaBoost with Metadata')
    plot_learning_curve(adab_best_model, X_train, y_train, title="Learning Curve for AdaBoost with Metadata", filename='adaboost_learning_curve_meta.png')

    # Get feature names from the column transformer
    adab_feature_names = numeric_features + \
        list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))

    # Calculate feature importances and display the top 10
    adab_feature_importances = adab_best_model.feature_importances_
    adab_importances_df = pd.DataFrame({
        'feature': adab_feature_names,
        'importance': adab_feature_importances
    })
    adab_importances_df = adab_importances_df.sort_values(by='importance', ascending=False)

    # Plotting the top 10 features
    plt.figure(figsize=(10, 6))
    adab_importances_df[:10].plot(kind='barh', x='feature', y='importance', legend=False, color='skyblue')
    plt.gca().invert_yaxis()
    plt.title('Top 10 Feature Importances in AdaBoost Model')
    plt.xlabel('Relative Importance')
    plt.ylabel('Features')
    plt.tight_layout()
    plt.show()

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
    svm_search = RandomizedSearchCV(svm_model, svm_param, n_iter=100, cv=cv, scoring='roc_auc', random_state=25, verbose=2, n_jobs=-1)

    # Train and evaluate model
    svm_predictions, svm_probabilities, svm_best_model = train_and_evaluate(svm_search, X_train, y_train, X_test)

    # Calculate metrics and plot
    calculate_and_print_metrics(y_test, svm_predictions, svm_probabilities, 'SVM with Metadata')
    plot_roc_curve(y_test, svm_probabilities, 'SVM with Metadata')
    plot_learning_curve(svm_best_model, X_train, y_train, title="Learning Curve for SVM with Metadata", filename='svm_learning_curve_meta.png')


#========================================================================================================================================#
#For execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a machine learning model on provided dataset.")
    parser.add_argument('filename', type=str, help="Path to the dataset file.")
    args = parser.parse_args()

    main(args)
