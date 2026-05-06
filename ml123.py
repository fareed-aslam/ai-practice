
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['MedHouseVal'] = data.target

print(df.info())
print(df.describe())

plt.scatter(df['MedInc'], df['MedHouseVal'])
plt.xlabel("Median Income")
plt.ylabel("House Value")
plt.title("Income vs House Price")
plt.show()


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()

X = df.drop('MedHouseVal', axis=1)
y = df['MedHouseVal']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
dt = DecisionTreeRegressor()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

print("Linear Regression MSE:", mean_squared_error(y_test, lr_pred))
print("Linear Regression R2:", r2_score(y_test, lr_pred))

print("Decision Tree MSE:", mean_squared_error(y_test, dt_pred))
print("Decision Tree R2:", r2_score(y_test, dt_pred))

plt.scatter(y_test, lr_pred)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Linear Regression")
plt.show()

-------q2-------

import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


df = sns.load_dataset('titanic')


print(df.info())
print(df.describe())

sns.countplot(x='survived', data=df)
plt.show()

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# HANDLE MISSING
df['age'].fillna(df['age'].mean(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# ENCODING
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

# FEATURES
X = df[['pclass','sex','age','fare','embarked']]
y = df['survived']

# SCALING
scaler = StandardScaler()
X = scaler.fit_transform(X)

# SPLIT
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)

# MODELS
lr = LogisticRegression()
dt = DecisionTreeClassifier()

lr.fit(X_train,y_train)
dt.fit(X_train,y_train)

# PREDICT
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

# EVALUATE
print("Logistic Accuracy:", accuracy_score(y_test, lr_pred))
print("Decision Tree Accuracy:", accuracy_score(y_test, dt_pred))

print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))
-------q3------

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# LOAD DATA
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

# EDA
print(df.info())
print(df.describe())

plt.scatter(df['sepal length (cm)'], df['petal length (cm)'])
plt.show()

sns.heatmap(df.corr(), annot=True)
plt.show()


scaler = StandardScaler()
X = scaler.fit_transform(df)

scores = []
k_range = range(2,10)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X)
    score = silhouette_score(X, labels)
    scores.append(score)


plt.plot(k_range, scores)
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.show()


best_k = k_range[scores.index(max(scores))]
print("Best K:", best_k)


kmeans = KMeans(n_clusters=best_k)
labels = kmeans.fit_predict(X)


plt.scatter(X[:,0], X[:,2], c=labels)
plt.scatter(kmeans.cluster_centers_[:,0],
            kmeans.cluster_centers_[:,2],
            c='red', marker='X')
plt.show()
