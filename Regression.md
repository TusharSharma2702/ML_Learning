# Regression
### What is Regression
* Regression is a statistical method in Machine Learning which tells relationship between a dependent variable (often called the target) and independent variables (predictors or features).
* The goal is to understand how dependent variable change with the change in dependent variable.

### Types of Regression
#### 1) Linear Regression
* Model that models a relationship  between variables using straight line, represented by formula<br>
        Y = mx+b<br>
where Y is the dependent variable, x is the independent variable, m is the slope, and b is the y-intercept.
* If have more than one independent variable than equation is, Y = m0+m1x1+m2x2+.....mnxn+ϵ
##### Sample code
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'X': [1, 2, 3, 4, 5],
    'Y': [2, 3, 5, 7, 11]
}

df = pd.DataFrame(data)

X = df[['X']]  
y = df['Y']    

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
```

#### 2) Logistic Regression
* Logistic regression is a statistical method used for binary classification problems, where the goal is to predict one of two possible outcomes based on one or more predictor variables.
* Used for binary outcomes (1/0, True/False).

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

data = {
    'X1': [2.5, 3.5, 5.5, 7.5, 8.5],
    'X2': [1.5, 1.0, 0.5, 0.1, 0.0],
    'Y': [0, 0, 1, 1, 1]  
}

df = pd.DataFrame(data)

X = df[['X1', 'X2']]  
y = df['Y']           


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model
model = LogisticRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')

# Optional: Visualize decision boundary (for 2D)
plt.scatter(df['X1'], df['X2'], c=df['Y'], cmap='coolwarm')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('Logistic Regression')
plt.show()
```