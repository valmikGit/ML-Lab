# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
import matplotlib.pyplot as plt
import seaborn as sns

# Load and clean the training data
train_data = pd.read_csv(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\IIITB Folder\5th Semester\Machine Learning\ML Lab\Assignment 2\train.csv')

# Advanced Feature Engineering
train_data['FamilySize'] = train_data['SibSp'] + train_data['Parch'] + 1
train_data['IsAlone'] = (train_data['FamilySize'] == 1).astype(int)

# Extract Title from Name
train_data['Title'] = train_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
train_data['Title'] = train_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
train_data['Title'] = train_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
train_data['Title'] = train_data['Title'].replace('Mme', 'Mrs')

# Create Age bins
train_data['Age_bin'] = pd.cut(train_data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=[0, 1, 2, 3, 4])
train_data['Fare_bin'] = pd.qcut(train_data['Fare'], 4, labels=[0, 1, 2, 3])

# Drop irrelevant columns
train_data = train_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Split the data into features and target variable
X = train_data.drop('Survived', axis=1)
y = train_data['Survived']

# Handle missing values with SimpleImputer
imputer = SimpleImputer(strategy='median')
X[['Age', 'Fare']] = imputer.fit_transform(X[['Age', 'Fare']])

# Fill missing values for Embarked with mode
X['Embarked'] = X['Embarked'].fillna(X['Embarked'].mode()[0])

# Convert categorical variables to numeric
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X = pd.get_dummies(X, columns=['Embarked', 'Pclass', 'Title', 'Age_bin', 'Fare_bin'], drop_first=True)

# Step 2: Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Hyperparameter tuning using GridSearchCV for Decision Tree
param_grid = {
    'max_depth': [3, 5, 7, 10, 12],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dtree = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dtree, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Get the best parameters from grid search
best_dtree = grid_search.best_estimator_

# Evaluate decision tree
y_pred_dtree = best_dtree.predict(X_val)
dtree_accuracy = accuracy_score(y_val, y_pred_dtree)
print(f'Best Decision Tree Accuracy: {dtree_accuracy:.4f}')

# Step 4: Random Forest and XGBoost Ensemble Models
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Fit models
rfc.fit(X_train, y_train)
xgb.fit(X_train, y_train)

# Predict and evaluate Random Forest
y_pred_rfc = rfc.predict(X_val)
rfc_accuracy = accuracy_score(y_val, y_pred_rfc)
print(f'Random Forest Accuracy: {rfc_accuracy:.4f}')

# Predict and evaluate XGBoost
y_pred_xgb = xgb.predict(X_val)
xgb_accuracy = accuracy_score(y_val, y_pred_xgb)
print(f'XGBoost Accuracy: {xgb_accuracy:.4f}')

# Step 5: Voting Classifier for ensemble method
voting_clf = VotingClassifier(estimators=[
    ('dtree', best_dtree),
    ('rfc', rfc),
    ('xgb', xgb)
], voting='soft')

# Fit the Voting Classifier
voting_clf.fit(X_train, y_train)

# Predict and evaluate
y_pred_voting = voting_clf.predict(X_val)
voting_accuracy = accuracy_score(y_val, y_pred_voting)
print(f'Voting Classifier Accuracy: {voting_accuracy:.4f}')

# Load and prepare the test data
test_data = pd.read_csv(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\IIITB Folder\5th Semester\Machine Learning\ML Lab\Assignment 2\test.csv')

# Apply the same feature engineering and preprocessing as the training data
test_data['FamilySize'] = test_data['SibSp'] + test_data['Parch'] + 1
test_data['IsAlone'] = (test_data['FamilySize'] == 1).astype(int)

test_data['Title'] = test_data['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
test_data['Title'] = test_data['Title'].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
test_data['Title'] = test_data['Title'].replace(['Mlle', 'Ms'], 'Miss')
test_data['Title'] = test_data['Title'].replace('Mme', 'Mrs')

# Create Age bins and Fare bins for test data
test_data['Age_bin'] = pd.cut(test_data['Age'], bins=[0, 12, 20, 40, 60, 80], labels=[0, 1, 2, 3, 4])
test_data['Fare_bin'] = pd.qcut(test_data['Fare'], 4, labels=[0, 1, 2, 3])

# Drop irrelevant columns
test_data = test_data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)

# Handle missing values for the test data
test_data[['Age', 'Fare']] = imputer.transform(test_data[['Age', 'Fare']])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])

# Convert categorical variables to numeric for test data
test_data['Sex'] = test_data['Sex'].map({'male': 0, 'female': 1})
test_data = pd.get_dummies(test_data, columns=['Embarked', 'Pclass', 'Title', 'Age_bin', 'Fare_bin'], drop_first=True)

# Ensure the test data has the same columns as the training set
missing_cols = set(X.columns) - set(test_data.columns)
for col in missing_cols:
    test_data[col] = 0

# Align the columns in the test set to match the training set
test_data = test_data[X.columns]

# Step 6: Predict on the test data using the Voting Classifier
test_predictions = voting_clf.predict(test_data)

# Prepare submission file
submission = pd.DataFrame({
    'PassengerId': pd.read_csv(r'C:\Users\Valmik Belgaonkar\OneDrive\Desktop\IIITB Folder\5th Semester\Machine Learning\ML Lab\Assignment 2\test.csv')['PassengerId'],
    'Survived': test_predictions
})
submission.to_csv('submission.csv', index=False)

print("Submission file created.")