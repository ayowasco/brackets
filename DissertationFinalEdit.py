#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[7]:


import pandas as pd

#To read the dataset.csv file
file_path = "C:\\Users\\AYOWASCO\\OneDrive - University of Stirling\\Desktop\\dataset.csv"
data = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(data.head())



# In[8]:


# Get a summary of the DataFrame
print(data.info())

# Generate descriptive statistics
print(data.describe())

# Check the column names
print(data.columns)

# Check the shape of the DataFrame
print(data.shape)


# In[10]:


# A bar plot can be used to show the distribution of the target variable,
# indicating how many students graduated versus dropped out.

import matplotlib.pyplot as plt
import seaborn as sns

# Ensure that the column name is correctly spelled
column_name = 'Target'  # Replace 'Target' with the actual column name if different

# Check if the column exists
if column_name in data.columns:
    # Count plot for the target variable
    plt.figure(figsize=(8, 6))
    sns.countplot(x=column_name, data=data)
    plt.title('Distribution of Student Outcomes')
    plt.xlabel('Outcome')
    plt.ylabel('Count')
    plt.show()
else:
    print(f"Column '{column_name}' not found in the dataset.")


# In[ ]:





# In[12]:


# A heatmap can be used to visualize the correlations between different features and
# identify potential predictors of student performance.

import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns from the DataFrame
numeric_data = data.select_dtypes(include=['number'])

# Calculate the correlation matrix for numeric data
corr_matrix = numeric_data.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap of Numeric Features')
plt.show()



# In[ ]:





# In[13]:


# For models like Random Forest, you can visualize feature 
# importance to understand which variables have the most influence on the predictions.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Splitting the data
X = data.drop('Target', axis=1)
y = data['Target'].apply(lambda x: 1 if x == 'Graduate' else 0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a random forest model
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)

# Get feature importances
feature_importances = pd.Series(rf.feature_importances_, index=X.columns)

# Plot feature importances
plt.figure(figsize=(10, 8))
feature_importances.nlargest(10).plot(kind='barh')
plt.title('Top 10 Important Features')
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.show()


# In[ ]:





# In[14]:


# An ROC curve is useful for evaluating the performance of the classification model 
# by visualizing the trade-off between sensitivity and specificity.

from sklearn.metrics import roc_curve, auc

# Predict probabilities
y_prob = rf.predict_proba(X_test)[:, 1]

# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


# A box plot is useful for visualizing the distribution of numerical data and  
# identifying outliers across different categories.

import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# Load the data from the provided path (assuming it is already loaded in this session)
# data = pd.read_csv('your_dataset.csv')  # Load your dataset if not already done

# Display column names to verify them
print("Columns in DataFrame:", data.columns)

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Verify if 'Admission grade' and 'Target' are the correct column names
if 'Admission grade' in data.columns and 'Target' in data.columns:
    # Box plot of admission grades by student outcome
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Target', y='Admission grade', data=data)
    plt.title('Box Plot of Admission Grades by Student Outcome')
    plt.xlabel('Outcome')
    plt.ylabel('Admission Grade')
    plt.show()
else:
    print("Columns 'Admission grade' or 'Target' not found. Please check for typos or incorrect naming.")


# In[ ]:





# In[26]:


# A scatter plot with a regression line can help visualize 
#  the relationship between two numerical variables, along with the line of best fit.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Display the column names to identify the correct ones
print("Columns in DataFrame:", list(data.columns))

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Display the first few rows of the DataFrame to verify the data
print(data.head())

# Check if 'Admission grade' and 'Curricular units 1st sem (approved)' are correct
# Adjust these variables to match the exact names
admission_col = 'Admission Grade'  # Adjust to the exact name found in data
units_col = 'Curricular Units 1st Sem (Approved)'  # Adjust to the exact name found in data

# Verify if the columns are correct and exist in the DataFrame
if admission_col in data.columns and units_col in data.columns:
    # Check for missing values in the relevant columns
    print(data[[admission_col, units_col]].isnull().sum())

    # Ensure there are no missing values in the selected columns
    data = data.dropna(subset=[admission_col, units_col])

    # Scatter plot with regression line
    plt.figure(figsize=(10, 6))
    sns.regplot(x=admission_col, y=units_col, data=data, scatter_kws={'alpha': 0.5})
    plt.title('Admission Grade vs. Curricular Units Approved (1st Sem)')
    plt.xlabel(admission_col)
    plt.ylabel(units_col)
    plt.show()
else:
    print(f"Columns '{admission_col}' or '{units_col}' not found. Please check for typos or incorrect naming.")



# In[ ]:





# In[27]:


# histogram to visualize the distribution of ages in the dataset


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming the data is already loaded into the 'data' DataFrame

# Display the column names to identify the correct ones
print("Columns in DataFrame:", list(data.columns))

# Strip any leading/trailing whitespace from column names
data.columns = data.columns.str.strip()

# Display the first few rows of the DataFrame to verify the data
print(data.head())

# Assuming the age column is correctly named as 'Age at enrollment'
age_col = 'Age at enrollment'  # Adjust to the exact name found in data

# Verify if the column exists
if age_col in data.columns:
    # Plot histogram of ages
    plt.figure(figsize=(10, 6))
    sns.histplot(data[age_col], bins=20, kde=True, color='skyblue')
    plt.title('Histogram of Ages at Enrollment')
    plt.xlabel('Age at Enrollment')
    plt.ylabel('Frequency')
    plt.show()
else:
    print(f"Column '{age_col}' not found. Please check for typos or incorrect naming.")


# In[ ]:





# In[7]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\\Users\\AYOWASCO\\OneDrive - University of Stirling\\Desktop\\dataset.csv"
data = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Overview:")
print(data.info())

# Summary statistics
print("\nSummary Statistics:")
print(data.describe())

# Distribution of categorical variables (example: Marital Status, Nationality)
print("\nCategorical Variable Distributions:")
print(data['Marital status'].value_counts())
print(data['Nationality'].value_counts())

# Histograms for numerical features (example: Grades, Curricular Units)
data[['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)', 
      'Unemployment rate', 'GDP']].hist(bins=20, figsize=(10, 8))
plt.show()

# Select only numeric columns for correlation matrix
numeric_data = data.select_dtypes(include=['float64', 'int64'])

# Correlation matrix
corr_matrix = numeric_data.corr()
plt.figure(figsize=(15, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
plt.title('Correlation Matrix of Features')
plt.show()

# Boxplot to check for outliers in Grades
plt.figure(figsize=(10, 6))
sns.boxplot(data=data[['Curricular units 1st sem (grade)', 'Curricular units 2nd sem (grade)']])
plt.title('Boxplot of Grades')
plt.show()


# In[ ]:





# In[8]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
file_path = "C:\\Users\\AYOWASCO\\OneDrive - University of Stirling\\Desktop\\dataset.csv"
data = pd.read_csv(file_path)

# Plotting histograms
plt.figure(figsize=(16, 10))

# Histogram for Age at Enrollment
plt.subplot(2, 3, 1)
plt.hist(data['Age at enrollment'], bins=20, color='purple', alpha=0.7)
plt.title('Histogram of Age at Enrollment')
plt.xlabel('Age at Enrollment')
plt.ylabel('Frequency')

# Histogram for Curricular units 1st sem (grade)
plt.subplot(2, 3, 2)
plt.hist(data['Curricular units 1st sem (grade)'], bins=20, color='blue', alpha=0.7)
plt.title('Histogram of Curricular units 1st sem (grade)')
plt.xlabel('Grade')
plt.ylabel('Frequency')

# Histogram for Curricular units 2nd sem (grade)
plt.subplot(2, 3, 3)
plt.hist(data['Curricular units 2nd sem (grade)'], bins=20, color='green', alpha=0.7)
plt.title('Histogram of Curricular units 2nd sem (grade)')
plt.xlabel('Grade')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# Plotting boxplots
plt.figure(figsize=(16, 6))

# Boxplot for Age at Enrollment
plt.subplot(1, 3, 1)
sns.boxplot(data['Age at enrollment'], color='lightblue')
plt.title('Boxplot of Age at Enrollment')
plt.xlabel('Age at Enrollment')

# Boxplot for Curricular units 1st sem (grade)
plt.subplot(1, 3, 2)
sns.boxplot(data['Curricular units 1st sem (grade)'], color='lightgreen')
plt.title('Boxplot of Curricular units 1st sem (grade)')
plt.xlabel('Grade')

# Boxplot for Curricular units 2nd sem (grade)
plt.subplot(1, 3, 3)
sns.boxplot(data['Curricular units 2nd sem (grade)'], color='lightcoral')
plt.title('Boxplot of Curricular units 2nd sem (grade)')
plt.xlabel('Grade')

plt.tight_layout()
plt.show()


# In[ ]:





# In[10]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Load the dataset
file_path = "C:\\Users\\AYOWASCO\\OneDrive - University of Stirling\\Desktop\\dataset.csv"
data = pd.read_csv(file_path)

# Calculate the correlation between age at enrollment and curricular unit grades
age_grade_corr_1st_sem, p_value_1st_sem = pearsonr(data['Age at enrollment'], data['Curricular units 1st sem (grade)'])
age_grade_corr_2nd_sem, p_value_2nd_sem = pearsonr(data['Age at enrollment'], data['Curricular units 2nd sem (grade)'])

print(f"Correlation between Age at Enrollment and 1st Sem Grade: {age_grade_corr_1st_sem:.2f} (p-value: {p_value_1st_sem:.4f})")
print(f"Correlation between Age at Enrollment and 2nd Sem Grade: {age_grade_corr_2nd_sem:.2f} (p-value: {p_value_2nd_sem:.4f})")

# Visualize the relationship
sns.lmplot(x='Age at enrollment', y='Curricular units 1st sem (grade)', data=data, aspect=1.5)
plt.title('Age at Enrollment vs. 1st Sem Grade')
plt.show()

sns.lmplot(x='Age at enrollment', y='Curricular units 2nd sem (grade)', data=data, aspect=1.5)
plt.title('Age at Enrollment vs. 2nd Sem Grade')
plt.show()


# In[ ]:





# In[25]:


print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")



# In[26]:


# Check the feature names in both datasets
print("Training features:", X_train.columns)
print("Test features:", X_test.columns)

# Identify the features that are in training but not in test, and vice versa
missing_in_test = set(X_train.columns) - set(X_test.columns)
missing_in_train = set(X_test.columns) - set(X_train.columns)

print("Features in training but not in test:", missing_in_test)
print("Features in test but not in training:", missing_in_train)


# In[27]:





# In[12]:


import pandas as pd
import numpy as np

# Model Libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# Model Evaluation
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler





# In[29]:





# In[16]:


# Load the dataset
data = pd.read_csv ("C:/Users/AYOWASCO/Downloads/engineered_dataset.csv")

# Identify features and target variable
X = data.drop('Target', axis=1)
y = data['Target']

# To handle missing values
print(data.isnull().sum())
X = X.fillna(X.median())

# To Perform feature engineering
# Example: Create new features, transform existing ones, or drop unneeded features
# For simplicity, let's assume we're scaling the features



scaler = StandardScaler()
X_engineered = scaler.fit_transform(X_raw)

# Save the engineered dataset for later use
engineered_data = data.DataFrame(X_engineered, columns=X_raw.columns)
engineered_data['Target'] = y

# Save the engineered dataset to CSV
engineered_data.to_csv('engineered_dataset.csv', index=False)

print("Engineered dataset created and saved.")


# In[ ]:





# In[14]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the dataset
#df = pd.read_csv('path_to_your_dataset.csv')

# Define the feature matrix (X) and target variable (y)
# Replace 'Target' with your actual target column name, e.g., 'Dropout'
X_raw = data.drop('Target', axis=1)  # Features (exclude the target column)
y = data['Target']  # Target variable (e.g., 'Dropout')

# Handle missing values in X_raw if necessary (fill with median, or another approach)
X_raw = X_raw.fillna(X_raw.median())

# Feature scaling (assuming numerical features)
scaler = StandardScaler()
X_engineered = scaler.fit_transform(X_raw)

# Save the engineered dataset for later use
engineered_data = pd.DataFrame(X_engineered, columns=X_raw.columns)
engineered_data['Target'] = y

# Save the engineered dataset to CSV
engineered_data.to_csv('engineered_dataset.csv', index=False)

print("Engineered dataset created and saved.")


# In[ ]:





# In[28]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Assuming you have already defined X and y
# For example:
X_raw = data.drop('Target', axis=1)  # Replace 'Target' with your actual target column
y = data['Target']

# Feature scaling (scale the features in X_raw)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Now perform the train-test split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# If you need X_train and X_test without scaling, you can split X_raw:
X_train, X_test, _, _ = train_test_split(X_raw, y, test_size=0.2, random_state=42, stratify=y)


# In[31]:


from sklearn.preprocessing import LabelEncoder

# Load the dataset
#df = pd.read_csv('path_to_your_dataset.csv')

# Define the feature matrix (X) and target variable (y)
X_raw = data.drop('Target', axis=1)  # Replace 'Target' with your actual target column
y = data['Target']  # Replace 'Target' with your actual target variable

# Encode the target variable (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Now perform the train-test split using the encoded target variable
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# If you also need X_train and X_test without scaling:
X_train, X_test, _, _ = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# In[32]:


# Evaluate all models
results = {}
for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# Display results
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[33]:


# Training the models
# Initialize models with default parameters
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

# Function to train and evaluate models
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted')
    recall = recall_score(y_te, y_pred, average='weighted')
    f1 = f1_score(y_te, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Evaluating the models
results = {}
for name, model in models.items():
    if name in ['SVM', 'MLP']:
        acc, prec, rec, f1 = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# To display results
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[21]:


from sklearn.preprocessing import LabelEncoder

# Encode the target variable (y)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Now use y_encoded instead of y for train-test split
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# If you also need X_train and X_test without scaling:
X_train, X_test, _, _ = train_test_split(X_raw, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)


# Evaluate all models
results = {}
for name, model in models.items():
    if name in ['SVM', 'MLP']:
        acc, prec, rec, f1 = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# Display results
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[ ]:





# In[34]:


# Hyperparameter tunning for Catboost model

catboost_param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

catboost_model = CatBoostClassifier(random_state=42, verbose=0)
catboost_grid = GridSearchCV(estimator=catboost_model,
                             param_grid=catboost_param_grid,
                             cv=3,
                             scoring='f1_weighted',
                             n_jobs=-1)
catboost_grid.fit(X_train, y_train)
catboost_best = catboost_grid.best_estimator_

print("Best Parameters for CatBoost:", catboost_grid.best_params_)
print("Best F1 Score for CatBoost:", catboost_grid.best_score_)

y_pred_catboost = catboost_best.predict(X_test)
catboost_test_f1 = f1_score(y_test, y_pred_catboost, average='weighted')
print("Test F1 Score for CatBoost:", catboost_test_f1)


# In[ ]:





# In[35]:


from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

# Define your parameter grid
catboost_param_grid = {
    'iterations': [100, 200],
    'learning_rate': [0.01, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5]
}

# Initialize CatBoostClassifier
catboost_model = CatBoostClassifier(random_state=42, verbose=0)

# Initialize GridSearchCV
catboost_grid = GridSearchCV(estimator=catboost_model,
                             param_grid=catboost_param_grid,
                             cv=3,
                             scoring='f1_weighted',
                             n_jobs=-1)

# Fit the model
catboost_grid.fit(X_train, y_train)

# Get the best model and its parameters
catboost_best = catboost_grid.best_estimator_

# Print the best parameters and F1 score
print("Best Parameters for CatBoost:", catboost_grid.best_params_)
print("Best F1 Score for CatBoost:", catboost_grid.best_score_)

# Predict and calculate F1 score on the test set
y_pred_catboost = catboost_best.predict(X_test)
catboost_test_f1 = f1_score(y_test, y_pred_catboost, average='weighted')
print("Test F1 Score for CatBoost:", catboost_test_f1)


# In[ ]:





# In[36]:


# Hyperparameter tunning for xgboost model

xgboost_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}

xgboost_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
xgboost_grid = GridSearchCV(estimator=xgboost_model,
                            param_grid=xgboost_param_grid,
                            cv=3,
                            scoring='f1_weighted',
                            n_jobs=-1)
xgboost_grid.fit(X_train, y_train)
xgboost_best = xgboost_grid.best_estimator_

print("Best Parameters for XGBoost:", xgboost_grid.best_params_)
print("Best F1 Score for XGBoost:", xgboost_grid.best_score_)

y_pred_xgboost = xgboost_best.predict(X_test)
xgboost_test_f1 = f1_score(y_test, y_pred_xgboost, average='weighted')
print("Test F1 Score for XGBoost:", xgboost_test_f1)


# In[ ]:





# In[37]:


# Hyperparameter tunning for lightgbm model

lightgbm_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.01, 0.1],
    'num_leaves': [31, 50],
    'max_depth': [4, 6, 8],
    'subsample': [0.7, 1],
    'colsample_bytree': [0.7, 1]
}

lightgbm_model = LGBMClassifier(random_state=42)
lightgbm_grid = GridSearchCV(estimator=lightgbm_model,
                             param_grid=lightgbm_param_grid,
                             cv=3,
                             scoring='f1_weighted',
                             n_jobs=-1)
lightgbm_grid.fit(X_train, y_train)
lightgbm_best = lightgbm_grid.best_estimator_

print("Best Parameters for LightGBM:", lightgbm_grid.best_params_)
print("Best F1 Score for LightGBM:", lightgbm_grid.best_score_)

y_pred_lightgbm = lightgbm_best.predict(X_test)
lightgbm_test_f1 = f1_score(y_test, y_pred_lightgbm, average='weighted')
print("Test F1 Score for LightGBM:", lightgbm_test_f1)


# In[ ]:





# In[38]:


# Confusion Matrix for CatBoost
conf_matrix = confusion_matrix(y_test, y_pred_catboost)
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - CatBoost')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[ ]:





# In[39]:


# Feature importance for CatBoost
feature_importances = catboost_best.get_feature_importance()
feature_names = X.columns
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False)
print(fi_df.head(10))


# In[ ]:





# In[40]:


# Evaluating the Tuned Models and Select the Best One
# Compile and compare final results for CatBoost, XGBoost, and LightGBM

tuned_results = {
    'CatBoost': {
        'Best CV F1 Score': catboost_grid.best_score_,
        'Test F1 Score': catboost_test_f1,
        'Best Parameters': catboost_grid.best_params_
    },
    'XGBoost': {
        'Best CV F1 Score': xgboost_grid.best_score_,
        'Test F1 Score': xgboost_test_f1,
        'Best Parameters': xgboost_grid.best_params_
    },
    'LightGBM': {
        'Best CV F1 Score': lightgbm_grid.best_score_,
        'Test F1 Score': lightgbm_test_f1,
        'Best Parameters': lightgbm_grid.best_params_
    }
}

tuned_results_df = pd.DataFrame(tuned_results).T
print(tuned_results_df)


# In[ ]:





# In[ ]:





# In[41]:


# Initialize models with default parameters
models = {
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
    'LightGBM': LGBMClassifier(random_state=42),
    'CatBoost': CatBoostClassifier(verbose=0, random_state=42),
    'SVM': SVC(probability=True, random_state=42),
    'MLP': MLPClassifier(random_state=42)
}

# Function to train and evaluate models
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted')
    recall = recall_score(y_te, y_pred, average='weighted')
    f1 = f1_score(y_te, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Dictionary to store results
results = {}

# Evaluate models
for name, model in models.items():
    if name in ['SVM', 'MLP']:
        acc, prec, rec, f1 = evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test)
    else:
        acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

    
    results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[44]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

# Set random seed for reproducibility
#random_state = 42

# Load the engineered dataset (assuming this is already preprocessed)
#df = pd.read_csv('path_to_your_engineered_dataset.csv')

# Replace whitespace in feature names
data.columns = data.columns.str.replace(' ', '_')

# Define features and target
X = data.drop('Target', axis=1)  # Adjust column name if different
y = data['Target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state, stratify=y)

# Initialize models with consistent random states
models = {
    'Random Forest': RandomForestClassifier(random_state=random_state),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
    'LightGBM': LGBMClassifier(random_state=random_state),
    'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0),
    'SVM': SVC(probability=True, random_state=random_state),
    'MLP': MLPClassifier(random_state=random_state)
}

# Function to evaluate models
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted')
    recall = recall_score(y_te, y_pred, average='weighted')
    f1 = f1_score(y_te, y_pred, average='weighted')
    return accuracy, precision, recall, f1

from sklearn.preprocessing import LabelEncoder

# Convert categorical target variable to numeric using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # This will encode 'Dropout', 'Enrolled', 'Graduate' to numeric values

# Split the data again, but use the encoded target variable
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded)

# Evaluate models with the encoded target
results = {}
for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# Convert results to DataFrame and display
results_df = pd.DataFrame(results).T
print(results_df.sort_values(by='F1 Score', ascending=False))



# Evaluate all models and store results
#results = {}
#for name, model in models.items():
 #   acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
  #  results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# Convert results to DataFrame
#results_df = pd.DataFrame(results).T
#print(results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[48]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
get_ipython().system('pip install shap')
import shap

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
random_state = 42

# Load the dataset (replace with the correct path to your dataset)
#df = pd.read_csv('path_to_your_engineered_dataset.csv')

# Replace whitespace in feature names with underscores for compatibility with LightGBM
data.columns = data.columns.str.replace(' ', '_')

# Define features and target
X = data.drop('Target', axis=1)  # Replace 'Target' with the actual column name if different
y = data['Target']

# Convert categorical target variable to numeric using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert 'Dropout', 'Enrolled', 'Graduate' to numeric values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=random_state, stratify=y_encoded)

# Initialize models with consistent random states
models = {
    'Random Forest': RandomForestClassifier(random_state=random_state),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_state),
    'LightGBM': LGBMClassifier(random_state=random_state),
    'CatBoost': CatBoostClassifier(random_state=random_state, verbose=0),
    'SVM': SVC(probability=True, random_state=random_state),
    'MLP': MLPClassifier(random_state=random_state)
}

# Function to train and evaluate models
def evaluate_model(model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    accuracy = accuracy_score(y_te, y_pred)
    precision = precision_score(y_te, y_pred, average='weighted')
    recall = recall_score(y_te, y_pred, average='weighted')
    f1 = f1_score(y_te, y_pred, average='weighted')
    return accuracy, precision, recall, f1

# Evaluate all models and store results
results = {}
for name, model in models.items():
    acc, prec, rec, f1 = evaluate_model(model, X_train, X_test, y_train, y_test)
    results[name] = {'Accuracy': acc, 'Precision': prec, 'Recall': rec, 'F1 Score': f1}

# Convert results to DataFrame
results_df = pd.DataFrame(results).T
print("Initial Model Performance:")
print(results_df.sort_values(by='F1 Score', ascending=False))

# Hyperparameter tuning for CatBoost using GridSearchCV
catboost_param_grid = {
    'iterations': [100, 200, 500],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5],
    'border_count': [32, 64, 128],
    'bagging_temperature': [0, 1, 2]
}

catboost_model = CatBoostClassifier(random_state=random_state, verbose=0)

# Use GridSearchCV to search for the best parameters
grid_search = GridSearchCV(estimator=catboost_model, param_grid=catboost_param_grid, cv=3, scoring='f1_weighted', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("\nBest Parameters for CatBoost:", grid_search.best_params_)
print("Best F1 Score for CatBoost:", grid_search.best_score_)

# Retrain CatBoost with the best parameters
catboost_best = grid_search.best_estimator_
catboost_best.fit(X_train, y_train)

# Evaluate CatBoost again after tuning
y_pred_catboost = catboost_best.predict(X_test)
catboost_acc = accuracy_score(y_test, y_pred_catboost)
catboost_prec = precision_score(y_test, y_pred_catboost, average='weighted')
catboost_rec = recall_score(y_test, y_pred_catboost, average='weighted')
catboost_f1 = f1_score(y_test, y_pred_catboost, average='weighted')

# Updated results after tuning
print("\nCatBoost Performance after Hyperparameter Tuning:")
print(f"Accuracy: {catboost_acc:.4f}, Precision: {catboost_prec:.4f}, Recall: {catboost_rec:.4f}, F1 Score: {catboost_f1:.4f}")

# Feature importance
feature_importances = catboost_best.get_feature_importance()
features = X_train.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("\nFeature Importance (Top 10):")
print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))

# SHAP analysis for CatBoost interpretability
explainer = shap.TreeExplainer(catboost_best)
shap_values = explainer.shap_values(X_test)

# Plot SHAP summary
shap.summary_plot(shap_values, X_test)

# Final comparison of CatBoost with other models
results['Tuned CatBoost'] = {'Accuracy': catboost_acc, 'Precision': catboost_prec, 'Recall': catboost_rec, 'F1 Score': catboost_f1}
final_results_df = pd.DataFrame(results).T
print("\nFinal Model Performance Comparison:")
print(final_results_df.sort_values(by='F1 Score', ascending=False))


# In[ ]:





# In[ ]:





# In[ ]:





# In[58]:


import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder

# Handle missing values for numerical columns
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

# Handle missing values for categorical columns
categorical_columns = data.select_dtypes(include=['object']).columns

for column in categorical_columns:
    mode_value = data[column].mode()
    if not mode_value.empty:
        data[column] = data[column].fillna(mode_value.iloc[0])

# Encode categorical variables
for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Define features and target
X = data.drop(columns=['Target'])  # Assuming 'Target' is the column for dropout prediction
y = data['Target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the CatBoost model with hyperparameter tuning
model = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.01,
    eval_metric='Accuracy',   # Use accuracy for early stopping
    custom_metric=['F1'],     # Track F1 score as a custom metric
    verbose=100
)

# Train the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, and recall
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")




# In[72]:


import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Preprocess the data (handle missing values, encode categorical variables)
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

categorical_columns = data.select_dtypes(include=['object']).columns
for column in categorical_columns:
    mode_value = data[column].mode()
    if not mode_value.empty:
        data[column] = data[column].fillna(mode_value.iloc[0])

for column in categorical_columns:
    data[column] = LabelEncoder().fit_transform(data[column])

# Define features and target
X = data.drop(columns=['Target'])  # Assuming 'Target' is the column for dropout prediction
y = data['Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the CatBoost model (use early stopping to shrink iterations if needed)
model = CatBoostClassifier(
    iterations=500,      # Shrink to 200 as per the table
    depth=5,             # Use depth from the table
    l2_leaf_reg=3,       # L2 regularization
    learning_rate=0.1,   # Learning rate
    eval_metric='Accuracy',  # Use 'Accuracy' or 'Logloss' for eval metric
    custom_metric=['F1'],    # Monitor F1 score
    verbose=100
)

# Train with early stopping
model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")



# In[18]:


from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Define features and target
X = data.drop(columns=['Target'])  # Assuming 'Target' is the column for dropout prediction
y = data['Target']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the parameter grid for RandomizedSearchCV
param_grid = {
    'iterations': [200, 500, 1000],
    'depth': [6, 7, 8, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'l2_leaf_reg': [1, 3, 5, 7],
    'border_count': [32, 50, 100],
    'custom_metric': ['F1']
}

# Create the CatBoostClassifier model
catboost_model = CatBoostClassifier(eval_metric='Accuracy', verbose=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=catboost_model,
    param_distributions=param_grid,
    n_iter=20,        # Number of different combinations to try
    scoring='accuracy',
    cv=3,             # 3-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1         # Use all available cores
)

# Fit the model
random_search.fit(X_train, y_train)

# Get the best parameters and train a final model
best_params = random_search.best_params_
print(f"Best Parameters: {best_params}")

# Train the final model with the best parameters
final_model = CatBoostClassifier(
    iterations=best_params['iterations'],
    depth=best_params['depth'],
    learning_rate=best_params['learning_rate'],
    l2_leaf_reg=best_params['l2_leaf_reg'],
    border_count=best_params['border_count'],
    eval_metric='Accuracy',
    custom_metric=['F1'],
    verbose=100
)

final_model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=50)

# Make predictions
y_pred = final_model.predict(X_test)

# Calculate accuracy, precision, recall, F1 score
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Output the results
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Precision: {precision * 100:.2f}%")
print(f"Recall: {recall * 100:.2f}%")
print(f"F1 Score: {f1 * 100:.2f}%")


# In[ ]:




