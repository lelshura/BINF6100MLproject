# randomly split a dataset into train and test sets
from sklearn.model_selection import train_test_split
import pandas as pd


# Define the location of the dataset
# In practice use argparse to read the file as input parameter
filename='rynazal_filtered_abundance.csv'

# Load the dataset; header is first row
df = pd.read_csv(filename, header=0)

data = df.values
# exclude "Sample ID" and response variable ("CRC") columns from features
X = data[:,1:-1]
y = data[:,-1]

# split into train test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.60, random_state=13, stratify=y)

print("X_train, X_test size: ", X_train.shape, X_test.shape)
print("y_train, y_test size: ", y_train.shape, y_test.shape)
print("\nX_train:\n", X_train, "\n")
print("X_test:\n", X_test, "\n")
print("y_train:\n", y_train, "\n")
print("y_test:\n", y_test, "\n")
