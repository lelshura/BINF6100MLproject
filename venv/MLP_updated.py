import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, \
    roc_curve
import matplotlib.pyplot as plt

# Define the location of the dataset
filename = '/Users/lubainakothari/Documents/BINF6100MLproject/rynazal_filtered_abundance.csv'

# Load the dataset; header is the first row
df = pd.read_csv(filename, header=0)

# Apply plot styling configurations
plt.rcParams.update({'font.size': 12})  # Set global font size
plt.rc('font', size=12)  # Apply font size to all text elements
plt.rc('axes', titlesize=16)  # Apply font size to the axes title
plt.rc('xtick', labelsize=12)  # Apply font size to the x-tick labels
plt.rc('ytick', labelsize=12)  # Apply font size to the y-tick labels

# Exclude 'Sample ID' and 'CRC' from the features, 'CRC' is the target variable
X = df.drop(['Sample ID', 'CRC'], axis=1).values
y = df['CRC'].astype(int).values  # Ensure the target is of integer type


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None, n_jobs=None,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    if ylim is not None:
        axes[0].set_ylim(*ylim)

    # Plot learning curve
    axes[0].set_title('Learning Curves')
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    # Plot scalability
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("Fit times")
    axes[1].set_title("Scalability of the model")

    # Plot fit times vs. score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("Fit times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")
    axes[2].set_ylim(*ylim)

    return plt

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, random_state=25, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define the MLP model
model = MLPClassifier(hidden_layer_sizes=(128, 64), activation='relu',
                      random_state=25, max_iter=1000, alpha=0.0001,
                      solver='adam', verbose=10, n_iter_no_change=10,
                      early_stopping=True, validation_fraction=0.1)

# Fit the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
y_probs = model.predict_proba(X_test)[:, 1]  # Get probabilities for the positive class
accuracy = accuracy_score(y_test, y_pred)
conf_mat = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_probs)
mcc = matthews_corrcoef(y_test, y_pred)

print('Accuracy: %.2f' % (accuracy * 100))
print('Confusion Matrix:\n', conf_mat)
print('ROC AUC Score: %.3f' % roc_auc)
print('Matthews Correlation Coefficient: %.3f' % mcc)
print('Classification Report:\n', classification_report(y_test, y_pred))

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_probs)
plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot the loss curve
plt.figure()
plt.plot(model.loss_curve_)
plt.title('Model Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()

# Configure the cross-validation strategy and plot learning curves
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
plot_learning_curve(model, "MLP Classifier", X_train, y_train, axes=axes, ylim=(0.5, 1.01), cv=cv, n_jobs=-1)
plt.show()