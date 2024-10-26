import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

path = "linearRegression/linearregression.csv"
df = pd.read_csv(path)


X = df[['x']]  
y = df['y']    

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f'Intercept: {model.intercept_}, Coefficient: {model.coef_}')

plt.scatter(X, y, color='blue')
plt.plot(X_test, predictions, color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.show()