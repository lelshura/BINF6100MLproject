import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Define the location of the dataset
filename = '/Users/lubainakothari/Documents/BINF6100MLproject/rynazal_filtered_abundance.csv'

# Load the dataset; header is the first row
df = pd.read_csv(filename, header=0)

# Exclude 'Sample ID' and 'CRC' from the features, 'CRC' is the target variable
X = df.drop(['Sample ID', 'CRC'], axis=1).values
y = df['CRC'].astype(int).values  # Ensure the target is of integer type

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
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %.2f' % (accuracy*100))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))

# Plot the loss curve
plt.plot(model.loss_curve_)
plt.title('Model Loss Curve')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.show()
