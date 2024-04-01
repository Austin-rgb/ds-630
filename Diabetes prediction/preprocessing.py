import pandas as pd

# Load the dataset
df = pd.read_csv('diabetes.csv')

# Display the first few rows of the dataset
print(df.head())

# Check the dimensions of the dataset
print("Shape of the dataset:", df.shape)

# Check for missing values
print("Missing values:\n", df.isnull().sum())

# Statistical summary of numerical features
print("Statistical summary:\n", df.describe())

# Replace missing values with mean
df.fillna(df.mean(), inplace=True)

from sklearn.model_selection import train_test_split

# Split the dataset into features and target variable
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split the dataset into training and testing sets (e.g., 70/30 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the training set
X_train_scaled = scaler.fit_transform(X_train)

# Transform the testing set
X_test_scaled = scaler.transform(X_test)
'''
from imblearn.over_sampling import SMOTE

# Initialize SMOTE for oversampling
smote = SMOTE(random_state=42)

# Resample the training data
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
'''