#!/usr/bin/env python
# coding: utf-8

# In[113]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv('file_location/WA_Fn-UseC_-HR-Employee-Attrition.csv')


# In[115]:


display(data)


# In[116]:


data.isnull().sum()


# In[117]:


data.shape


# In[118]:


#small dataset => use ML algorithms instead of deep learning 


# In[119]:


#Discover the dataset


# In[120]:


data.dtypes.value_counts()


# In[121]:


data1 = data.select_dtypes(include=['object'])
display(data1.head())


# In[122]:


for column in data1.columns:
    unique_values = data1[column].unique()
    print(f"values of column '{column}':\n{unique_values}\n")


# In[123]:


# Define mappings for specific columns
column_mappings = {
    'Attrition': {'No': 0, 'Yes': 1},
    'BusinessTravel': {'Travel_Frequently': 2, 'Travel_Rarely': 1, 'Non-Travel': 0},
    'Department': {'Research & Development': 3, 'Sales': 2, 'Human Resources': 1},
    'EducationField': {
        'Life Sciences': 1, 'Other': 6, 'Medical': 2, 'Marketing': 3,
        'Technical Degree': 4, 'Human Resources': 5
    },
    'Gender': {'Male': 0, 'Female': 1},
    'JobRole': {
        'Research Scientist': 1, 'Laboratory Technician': 2, 'Manufacturing Director': 3,
        'Healthcare Representative': 4, 'Manager': 5, 'Sales Representative': 6,
        'Research Director': 7, 'Sales Executive': 8, 'Human Resources': 9
    },
    'MaritalStatus': {'Married': 2, 'Single': 1, 'Divorced': 3},
    'Over18': {'Y': 1},
    'OverTime': {'No': 0, 'Yes': 1}
}


# In[124]:


# Use the replace method with the defined mappings for specific columns
data.replace(column_mappings, inplace=True)

# Print the updated DataFrame
display(data.head())


# In[125]:


# Split data into features and target.  Attrition is our target


# In[126]:


data_y= data["Attrition"]
data_x = data.drop("Attrition", axis=1)


# In[127]:


data_y


# In[ ]:





# In[128]:


data_x = data.drop("Attrition", axis=1)
data_x


# In[129]:


#we have some columns that contains only one unique value (e.g., all values are 1), it is a good choice to drop these columns.


# In[130]:


# Capture the list of columns to be dropped
dropped_columns = data_x.columns[data_x.nunique() == 1]

# Print the list of dropped columns
print("Dropped Columns:", dropped_columns)


# In[131]:


# Drop the constant columns and update the DataFrame
data_x = data_x.loc[:, data.nunique() > 1]


# In[ ]:





# In[132]:


from scipy.stats import skew


# In[133]:


def determine_scaling_method(column):
    # Calculate skewness of the feature
    skewness = skew(column)

    # Check skewness and range of values
    if abs(skewness) > 1.0:
        return "Standardization"  # Feature is significantly skewed
    elif (column.max() - column.min()) > 1:
        return "Normalization"  # Feature has a wide range
    else:
        return "No Scaling Required"


# In[134]:


# Loop through each column and determine scaling method
for column in data_x.columns:
    scaling_method = determine_scaling_method(data_x[column])
    print(f"Feature '{column}' should use: {scaling_method}")


# In[135]:


def scale_feature(column):
    # Calculate skewness of the feature
    skewness = skew(column)

    # Check skewness and range of values
    if abs(skewness) > 1.0:
        # Apply standardization
        scaler = StandardScaler()
        return scaler.fit_transform(column.values.reshape(-1, 1))
    elif (column.max() - column.min()) > 1:
        # Apply normalization
        scaler = MinMaxScaler()
        return scaler.fit_transform(column.values.reshape(-1, 1))
    else:
        # No scaling required
        return column.values


# In[136]:


for column in data_x.columns:
    data_x[column] = scale_feature(data_x[column])


# In[137]:


# Display the DataFrame with scaled columns
display(data_x.head())


# In[138]:


data_x.to_csv('scaled_x_data.csv', index=False)
data_y.to_csv('y_data.csv', index=False)


# In[ ]:


#####################################


# In[111]:


data.hist(figsize=(15,15))
plt.tight_layout()
plt.show


# In[15]:


data_attrition_yes=data[data["Attrition"]=="Yes"]


# In[17]:


data_attrition_yes.hist(figsize=(15,15))
plt.tight_layout()
plt.show


# In[ ]:




