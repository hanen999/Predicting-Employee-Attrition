#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import classification_report, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
# In[17]:


x=pd.read_csv('/mnt/scaled_x_data.csv')
y=pd.read_csv('/mnt/y_data.csv')


# #Data Splitting: Since the dataset is limited in size., an 80/20 split would be reasonable 



# In[77]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[32]:


#Model Selection: Choose at least two different predictive models 


# =>binary classification tasks:
# Logistic Regression: 
# Random Forest:
# Gradient Boosting: 
# Support Vector Machines (SVM): 
# Naive Bayes: 
# K-Nearest Neighbors (KNN): 
# Decision Trees:
# Neural Networks: Deep learning models
# 
# Random Forest:
# 
# -Random Forest is an ensemble learning technique that can work well with mixed data types (numeric and non-numeric).
# It can handle non-linear relationships in the data.
# Random Forest is less prone to overfitting compared to individual decision trees.
# It's a versatile and robust model that often provides good out-of-the-box performance.
# You can handle feature importance analysis easily with Random Forest.
# Gradient Boosting (e.g., XGBoost or LightGBM):
# 
# -Gradient boosting methods like XGBoost and LightGBM are powerful ensemble techniques that can handle non-linearity and mixed data types.
# They are often among the top performers in data science competitions.
# These models can be fine-tuned to find the best model parameters for your specific dataset.
# Gradient boosting models are robust and can handle small to medium-sized datasets effectively.
# 
# *
# In this particular case, Logistic Regression, Decision Trees, Naive Bayes, and K-Nearest Neighbors (KNN) might not be the best options for binary classification due to the following reasons:
# 
# Logistic Regression:
# 
# Logistic Regression assumes linear relationships between features and the log-odds of the target variable. It may not perform well when dealing with complex, non-linear relationships among your 25 features, especially if many of them are non-linear.
# Decision Trees:
# 
# Decision Trees can be prone to overfitting when the dataset is small and features are numerous, potentially leading to poor generalization.
# Naive Bayes:
# 
# Naive Bayes is a simple probabilistic classifier that assumes feature independence. It may not capture the intricate relationships and dependencies among your features, which can be crucial for accurate classification.
# K-Nearest Neighbors (KNN):
# 
# KNN relies on distances between data points, and it can struggle with high-dimensional datasets like the one with 25 features. The curse of dimensionality can lead to inefficiency and reduced performance.
# 
# Neural Networks:
# Neural networks are not suitable for this case due to the small dataset size, which can lead to overfitting.

# In[33]:


#Model Training: 


# In[78]:




# Create a Random Forest classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42) 

# Train the model on the training data
rf_model.fit(x_train, y_train)


# Create an XGBoost classifier
xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=42)  

# Train the model on the training data
xgb_model.fit(x_train, y_train)


# In[35]:


#Models Evaluation: 


# In[36]:




# In[79]:


# Predictions from Random Forest model
rf_predictions = rf_model.predict(x_test)

# Calculate evaluation metrics for Random Forest
rf_accuracy = accuracy_score(y_test, rf_predictions)
rf_precision = precision_score(y_test, rf_predictions)
rf_recall = recall_score(y_test, rf_predictions)
rf_f1 = f1_score(y_test, rf_predictions)
rf_roc_auc = roc_auc_score(y_test, rf_predictions)
rf_confusion = confusion_matrix(y_test, rf_predictions)

print("Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy}")
print(f"Precision: {rf_precision}")
print(f"Recall: {rf_recall}")
print(f"F1-Score: {rf_f1}")
print(f"ROC AUC: {rf_roc_auc}")
print(f"Confusion Matrix:\n{rf_confusion}")



# Predictions from XGBoost model
xgb_predictions = xgb_model.predict(x_test)

# Calculate evaluation metrics for XGBoost
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
xgb_precision = precision_score(y_test, xgb_predictions)
xgb_recall = recall_score(y_test, xgb_predictions)
xgb_f1 = f1_score(y_test, xgb_predictions)
xgb_roc_auc = roc_auc_score(y_test, xgb_predictions)
xgb_confusion = confusion_matrix(y_test, xgb_predictions)

print("\nXGBoost Metrics:")
print(f"Accuracy: {xgb_accuracy}")
print(f"Precision: {xgb_precision}")
print(f"Recall: {xgb_recall}")
print(f"F1-Score: {xgb_f1}")
print(f"ROC AUC: {xgb_roc_auc}")
print(f"Confusion Matrix:\n{xgb_confusion}")


# In[ ]:


##########"try dt"


# In[54]:



# Create a Decision Tree classifier
dt_model = DecisionTreeClassifier(random_state=42)

# Define a grid of hyperparameters to search
param_grid_dt = {
    'max_depth': [None, 5, 10, 20],   # Vary the maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Vary the minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Vary the minimum samples required for a leaf node
}

# Perform grid search with cross-validation
grid_search_dt = GridSearchCV(dt_model, param_grid_dt, cv=5, scoring='recall', n_jobs=-1)

# Fit the grid search to the data
grid_search_dt.fit(x_train, y_train)

# Get the best hyperparameters
best_params_dt = grid_search_dt.best_params_
best_dt_model = grid_search_dt.best_estimator_

# Train the best model on the training data
best_dt_model.fit(x_train, y_train)

# Make predictions and evaluate the best Decision Tree model
dt_predictions_tuned = best_dt_model.predict(x_test)

# Calculate evaluation metrics for the tuned Decision Tree
dt_accuracy_tuned = accuracy_score(y_test, dt_predictions_tuned)
dt_precision_tuned = precision_score(y_test, dt_predictions_tuned)
dt_recall_tuned = recall_score(y_test, dt_predictions_tuned)
dt_f1_tuned = f1_score(y_test, dt_predictions_tuned)
dt_roc_auc_tuned = roc_auc_score(y_test, dt_predictions_tuned)
dt_confusion_tuned = confusion_matrix(y_test, dt_predictions_tuned)

# Print or use the metrics as needed
print(f"Best Decision Tree Hyperparameters: {best_params_dt}")
print("Tuned Decision Tree Metrics:")
print(f"Accuracy: {dt_accuracy_tuned}")
print(f"Precision: {dt_precision_tuned}")
print(f"Recall: {dt_recall_tuned}")
print(f"F1-Score: {dt_f1_tuned}")
print(f"ROC AUC: {dt_roc_auc_tuned}")
print(f"Confusion Matrix:\n{dt_confusion_tuned}")


# In[ ]:


#Hyperparameter Tuning: 


# In[63]:



# Define a grid of hyperparameters to search
param_grid_rf = {
    'n_estimators': [3, 5, 10, 20, 30],  # Vary the number of trees
    'max_depth': [None, 5, 10, 20],   # Vary the maximum depth of trees
    'min_samples_split': [1, 2, 5, 10],  # Vary the minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4]  # Vary the minimum samples required for a leaf node
}

# Perform grid search with cross-validation
grid_search_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='recall', n_jobs=-1)

# Fit the grid search to the data
grid_search_rf.fit(x_train, y_train)

# Get the best hyperparameters
best_params_rf = grid_search_rf.best_params_
best_rf_model = grid_search_rf.best_estimator_

# Train the best model on the training data
best_rf_model.fit(x_train, y_train)

# Make predictions and evaluate the best model
rf_predictions_tuned = best_rf_model.predict(x_test)


# In[64]:


# Predictions from tuned Random Forest model
rf_predictions_tuned = best_rf_model.predict(x_test)

# Calculate evaluation metrics for tuned Random Forest
rf_accuracy_tuned = accuracy_score(y_test, rf_predictions_tuned)
rf_precision_tuned = precision_score(y_test, rf_predictions_tuned)
rf_recall_tuned = recall_score(y_test, rf_predictions_tuned)
rf_f1_tuned = f1_score(y_test, rf_predictions_tuned)
rf_roc_auc_tuned = roc_auc_score(y_test, rf_predictions_tuned)
rf_confusion_tuned = confusion_matrix(y_test, rf_predictions_tuned)

# Print or use the metrics as needed
print(f" best_params_rf: {best_params_rf} ")
print("Tuned Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy_tuned}")
print(f"Precision: {rf_precision_tuned}")
print(f"Recall: {rf_recall_tuned}")
print(f"F1-Score: {rf_f1_tuned}")
print(f"ROC AUC: {rf_roc_auc_tuned}")
print(f"Confusion Matrix:\n{rf_confusion_tuned}")


# In[51]:


# Predictions from tuned Random Forest model
rf_predictions_tuned = best_rf_model.predict(x_test)

# Calculate evaluation metrics for tuned Random Forest
rf_accuracy_tuned = accuracy_score(y_test, rf_predictions_tuned)
rf_precision_tuned = precision_score(y_test, rf_predictions_tuned)
rf_recall_tuned = recall_score(y_test, rf_predictions_tuned)
rf_f1_tuned = f1_score(y_test, rf_predictions_tuned)
rf_roc_auc_tuned = roc_auc_score(y_test, rf_predictions_tuned)
rf_confusion_tuned = confusion_matrix(y_test, rf_predictions_tuned)

# Print or use the metrics as needed
print(f" best_params_rf: {best_params_rf} ")
print("Tuned Random Forest Metrics:")
print(f"Accuracy: {rf_accuracy_tuned}")
print(f"Precision: {rf_precision_tuned}")
print(f"Recall: {rf_recall_tuned}")
print(f"F1-Score: {rf_f1_tuned}")
print(f"ROC AUC: {rf_roc_auc_tuned}")
print(f"Confusion Matrix:\n{rf_confusion_tuned}")


# In[ ]:





# In[61]:


# Define a grid of hyperparameters for XGBoost
param_grid_xgb = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 4, 5,10, 20],
    'learning_rate': [0.01, 0.1, 0.2],
    'min_child_weight': [1, 2, 3]
}


# Perform grid search with cross-validation
grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='recall', n_jobs=-1)

# Fit the grid search to the data
grid_search_xgb.fit(x_train, y_train)

# Get the best hyperparameters
best_params_xgb = grid_search_xgb.best_params_
best_xgb_model = grid_search_xgb.best_estimator_

# Train the best model on the training data
best_xgb_model.fit(x_train, y_train)

# Make predictions and evaluate the best model
xgb_predictions_tuned = best_xgb_model.predict(x_test)


# In[62]:


# Predictions from tuned XGBoost model
xgb_predictions_tuned = best_xgb_model.predict(x_test)

# Calculate evaluation metrics for tuned XGBoost
xgb_accuracy_tuned = accuracy_score(y_test, xgb_predictions_tuned)
xgb_precision_tuned = precision_score(y_test, xgb_predictions_tuned)
xgb_recall_tuned = recall_score(y_test, xgb_predictions_tuned)
xgb_f1_tuned = f1_score(y_test, xgb_predictions_tuned)
xgb_roc_auc_tuned = roc_auc_score(y_test, xgb_predictions_tuned)
xgb_confusion_tuned = confusion_matrix(y_test, xgb_predictions_tuned)

# Print or use the metrics as needed
print(f" best_params_rf: {best_params_xgb} ")
print("Tuned XGBoost Metrics:")
print(f"Accuracy: {xgb_accuracy_tuned}")
print(f"Precision: {xgb_precision_tuned}")
print(f"Recall: {xgb_recall_tuned}")
print(f"F1-Score: {xgb_f1_tuned}")
print(f"ROC AUC: {xgb_roc_auc_tuned}")
print(f"Confusion Matrix:\n{xgb_confusion_tuned}")


# In[ ]:


################################""


# In[14]:


importances = best_rf_model.feature_importances_

# Pair feature names with their importance scores
feature_importance = list(zip(x.columns, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Print or analyze feature importance
for feature, importance in feature_importance:
    print(f"{feature}: {importance}")


# In[15]:


importances = best_xgb_model.feature_importances_

# Pair feature names with their importance scores
feature_importance = list(zip(x.columns, importances))
feature_importance.sort(key=lambda x: x[1], reverse=True)

# Print or analyze feature importance
for feature, importance in feature_importance:
    print(f"{feature}: {importance}")


# In[13]:


feature_importance = best_rf_model.feature_importances_
feature_importance_df = pd.DataFrame({'Feature': x_train.columns, 'Importance': feature_importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)


# In[ ]:


############################################ hethom zeidin just kifeh sensitivity tt7sb mais deja 7asbin el recall w c mm


# In[65]:



# Calculate the confusion matrix for Random Forest (tuned)
rf_confusion_tuned = confusion_matrix(y_test, rf_predictions_tuned)

# Extract values from the confusion matrix
true_positives_rf = rf_confusion_tuned[1, 1]  # True Positives
false_negatives_rf = rf_confusion_tuned[1, 0]  # False Negatives

# Calculate Sensitivity (Recall) for Random Forest
sensitivity_rf = true_positives_rf / (true_positives_rf + false_negatives_rf)

print(f"Sensitivity (Recall) for Random Forest: {sensitivity_rf}")



# In[19]:


# Calculate the confusion matrix for XGBoost (tuned)
xgb_confusion_tuned = confusion_matrix(y_test, xgb_predictions_tuned)

# Extract values from the confusion matrix
true_positives_xgb = xgb_confusion_tuned[1, 1]  # True Positives
false_negatives_xgb = xgb_confusion_tuned[1, 0]  # False Negatives

# Calculate Sensitivity (Recall) for XGBoost
sensitivity_xgb = true_positives_xgb / (true_positives_xgb + false_negatives_xgb)

print(f"Sensitivity (Recall) for XGBoost: {sensitivity_xgb}")


# ###############  adjust threshold: If the probability score for an example is greater than or equal to the threshold, the model classifies it as the positive class; otherwise, it's classified as the negative class.

# In[75]:


# Example for Random Forest (tuned):

# Function to adjust threshold and calculate sensitivity
def adjust_threshold_and_evaluate(model, x, y, threshold):
    y_prob = model.predict_proba(x)[:, 1]  # Get probability scores for the positive class
    y_pred = (y_prob > threshold).astype(int)  # Adjust threshold for class prediction
    
    sensitivity = recall_score(y, y_pred)
    return sensitivity, y_pred

# Define a range of threshold values to test
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
sensitivity_scores = []
y_preds = []
# Iterate through different threshold values
for threshold in threshold_values:
    sensitivity, y_pred = adjust_threshold_and_evaluate(best_rf_model, x_test, y_test, threshold)
    sensitivity_scores.append(sensitivity)
    y_preds.append(y_pred)

# Find the threshold that maximizes sensitivity
best_threshold = threshold_values[sensitivity_scores.index(max(sensitivity_scores))]
print(f"Best Threshold for Max Sensitivity: {best_threshold}")

# Calculate sensitivity for the best threshold
best_sensitivity = max(sensitivity_scores)
print(f"Max Sensitivity: {best_sensitivity}")

# Evaluate other metrics for the model with the best threshold
best_y_pred = y_preds[sensitivity_scores.index(max(sensitivity_scores))]
print("Classification Report for Model with Max Sensitivity:")
print(classification_report(y_test, best_y_pred))


# In[81]:


# Example for Random Forest (not_tuned):

sensitivity_scores = []
y_preds = []
# Iterate through different threshold values
for threshold in threshold_values:
    sensitivity, y_pred = adjust_threshold_and_evaluate(rf_model, x_test, y_test, threshold)
    sensitivity_scores.append(sensitivity)
    y_preds.append(y_pred)

best_threshold = threshold_values[sensitivity_scores.index(max(sensitivity_scores))]
print(f"Best Threshold for Max Sensitivity: {best_threshold}")
best_sensitivity = max(sensitivity_scores)
print(f"Max Sensitivity: {best_sensitivity}")
best_y_pred = y_preds[sensitivity_scores.index(max(sensitivity_scores))]
print("Classification Report for Model with Max Sensitivity:")
print(classification_report(y_test, best_y_pred))


# In[69]:


# Example for XGBoost (tuned):


# Replace 'your_xgb_model' with your actual XGBoost model
your_xgb_model = best_xgb_model

# Define a range of threshold values to test
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

sensitivity_scores = []
y_preds = []

# Iterate through different threshold values
for threshold in threshold_values:
    sensitivity, y_pred = adjust_threshold_and_evaluate(your_xgb_model, x_test, y_test, threshold)
    sensitivity_scores.append(sensitivity)
    y_preds.append(y_pred)

# Find the threshold that maximizes sensitivity
best_threshold = threshold_values[sensitivity_scores.index(max(sensitivity_scores))]
print(f"Best Threshold for Max Sensitivity: {best_threshold}")

# Calculate sensitivity for the best threshold
best_sensitivity = max(sensitivity_scores)
print(f"Max Sensitivity: {best_sensitivity}")

# Evaluate other metrics for the model with the best threshold
best_y_pred = y_preds[sensitivity_scores.index(max(sensitivity_scores))]
print("Classification Report for Model with Max Sensitivity:")
print(classification_report(y_test, best_y_pred))

# Plot ROC curve and calculate ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, your_xgb_model.predict_proba(x_test)[:, 1])
roc_auc = roc_auc_score(y_test, your_xgb_model.predict_proba(x_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Corrected syntax here
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# In[70]:


# Example for XGBoost (not_tuned):


# Function to adjust threshold and calculate sensitivity
def adjust_threshold_and_evaluate(model, x, y, threshold):
    y_prob = model.predict_proba(x)[:, 1]  # Get probability scores for the positive class
    y_pred = (y_prob > threshold).astype(int)  # Adjust threshold for class prediction
    
    sensitivity = recall_score(y, y_pred)
    return sensitivity, y_pred

# Replace 'your_xgb_model' with your actual XGBoost model
your_xgb_model = xgb_model

# Define a range of threshold values to test
threshold_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

sensitivity_scores = []
y_preds = []

# Iterate through different threshold values
for threshold in threshold_values:
    sensitivity, y_pred = adjust_threshold_and_evaluate(your_xgb_model, x_test, y_test, threshold)
    sensitivity_scores.append(sensitivity)
    y_preds.append(y_pred)

# Find the threshold that maximizes sensitivity
best_threshold = threshold_values[sensitivity_scores.index(max(sensitivity_scores))]
print(f"Best Threshold for Max Sensitivity: {best_threshold}")

# Calculate sensitivity for the best threshold
best_sensitivity = max(sensitivity_scores)
print(f"Max Sensitivity: {best_sensitivity}")

# Evaluate other metrics for the model with the best threshold
best_y_pred = y_preds[sensitivity_scores.index(max(sensitivity_scores))]
print("Classification Report for Model with Max Sensitivity:")
print(classification_report(y_test, best_y_pred))

# Plot ROC curve and calculate ROC AUC
fpr, tpr, thresholds = roc_curve(y_test, your_xgb_model.predict_proba(x_test)[:, 1])
roc_auc = roc_auc_score(y_test, your_xgb_model.predict_proba(x_test)[:, 1])

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Corrected syntax here
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()


# #Sensitivity (Recall): The primary goal of the attrition prediction model is to identify employees at risk of attrition so that proactive interventions can be taken. High sensitivity ensures that the model can correctly identify as many employees at risk as possible, minimizing false negatives. In this context, false negatives (not identifying employees who are actually at risk) can be costly to the company, as it may result in attrition that could have been prevented. Prioritizing sensitivity helps in addressing this issue effectively.

# In[ ]:




