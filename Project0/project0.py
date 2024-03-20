import pandas as pd
from sklearn import linear_model

#------Training linear regression model
#DATA READING
data_train= pd.read_csv('train.csv', sep=',')
data_test= pd.read_csv('test.csv', sep=',')

X_train = data_train.drop(labels=['Id', 'y'], axis = 1)
y_train = data_train['y']

print(type(X_train))
# Instantiating LinearRegression() Model
lr = linear_model.LinearRegression()

# Training/Fitting the Model
lr.fit(X_train, y_train)

# At this point the coefficents are deteremened
# The coefficients
print("Coefficients: \n", lr.coef_)

#Formatting test data
X_test = data_test.drop(labels=['Id'], axis = 'columns')

# Making Predictions
pred = lr.predict(X_test)

print('Predictions: \n', pred)

# Create a DataFrame from the pred array
predictions = pd.DataFrame(pred, columns = ['y'])

# Add the 'Id' column from the data_test DataFrame
predictions['Id'] = data_test['Id']

# Make sure 'Id' is the first column
predictions = predictions[['Id', 'y']]

# Now save the DataFrame to a CSV file
predictions.to_csv('results.csv', index = False)