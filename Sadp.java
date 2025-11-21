1) Create a multiple linear regression model for house price dataset divide dataset into train and test data
   while giving it to model and predict prices of house.

   import pandas as pd
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_absolute_error

   data=pd.read_csv("house.csv")
   print(data)

   x=data[["bedrooms","sqft_living"]]
   y=data.price

   print(x)
   print(y)


   xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2)
   print(xtrain)
   print(xtest)
   print(ytrain)
   print(ytest)

   lr=LinearRegression()
   lr.fit(xtrain,ytrain)

   print(lr.intercept_)
   print(lr.coef_)

   print(lr.predict([[2,1000]]))


   ypred=lr.predict(xtest)
   cm=mean_absolute_error(ytest,ypred)
   print(cm)
2) Use dataset crash.csv is an accident survivor’s dataset portal for USA hosted by data.gov. The dataset
   contains passengers age and speed of vehicle (mph) at the time of impact and fate of passengers (1 for
   survived and 0 for not survived) after a crash. use logistic regression to decide if the age and speed
   can predict the survivability of the passengers.


import pandas as pd
data = pd.read_csv("crash.csv")
x = data[['age', 'speed']]
y = data['survived']
print(x)
print(y)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')


conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
coefficients = model.coef_
intercept = model.intercept_
print(f'Coefficients: {coefficients}')
print(f'Intercept: {intercept}')
3) Fit the simple linear regression and polynomial linear regression models to Salary_positions.csv data.
   Find which one is more accurately fitting to the given data. Also predict the salaries of level 11 and
   level 12 employees


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data=pd.read_csv("position_salaries.csv")
print(data)

x=data.iloc[:,1:2].values
y=data.iloc[:,2].values

print(x)
print(y)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25)

lr=LinearRegression()
lr.fit(xtrain,ytrain)
ypred=lr.predict(xtest)

plt.scatter(x,y,c="red")
plt.plot(x,lr.predict(x),c="green")
plt.show()

from sklearn.preprocessing import PolynomialFeatures
pr=PolynomialFeatures(degree=4)
xpoly=pr.fit_transform(x)
poreg=LinearRegression()
poreg.fit(xpoly,y)

plt.scatter(x,y,c="red")
plt.plot(x,poreg.predict(pr.fit_transform(x)),c="green")
plt.show()


print(lr.predict([[11]]))
print(poreg.predict(pr.fit_transform([[11]])))

print(lr.predict([[12]]))
print(poreg.predict(pr.fit_transform([[12]])))
4) Implement Ridge Regression, Lasso regression, ElasticNet model usingboston_houses.csv and take
   only ‘RM’ and ‘Price’ of the houses. divide the data as training and testing data. Fit line using Ridge
   regression and to find price of a house if it contains 5 rooms. and compare results.


import pandas as pd
data = pd.read_csv('boston_houses.csv')
data = data[['rm', 'price']]


X = data[['rm']]
y = data['price']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)


from sklearn.linear_model import Ridge
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)
price_5_rooms_ridge = ridge_model.predict([[5]])
print(" ridge Price for a house with 5 rooms:=>",price_5_rooms_ridge)


from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(X_train, y_train)
price_5_rooms_lasso = lasso_model.predict([[5]])
print("lasso Price for a house with 5 rooms: =>",price_5_rooms_lasso)


from sklearn.linear_model import ElasticNet
elasticnet_model = ElasticNet(alpha=1.0, l1_ratio=0.5)
elasticnet_model.fit(X_train, y_train)
price_5_rooms_elasticnet = elasticnet_model.predict([[5]])
print("ElasticNet Regression - Price for a house with 5 rooms: =>",price_5_rooms_elasticnet)
5) Write a python program to Implement Decision Tree classifier model onData which is extracted from
   images that were taken from genuine and forged banknote-like specimens.

            (refer UCI dataset https://archive.ics.uci.edu/dataset/267/banknote+authentication)




import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


data = pd.read_csv('banknote_data.csv')

x = data[['variance', 'skewness', 'kurtosis', 'entropy']]
y = data['class']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

dt_classifier = DecisionTreeClassifier(random_state=42)

dt_classifier.fit(X_train, y_train)

y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%\n")

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

from sklearn.tree import export_text

tree_rules = export_text(dt_classifier, feature_names=list(X.columns))
print("\nDecision Tree Rules:\n")
print(tree_rules)
6) Classify the iris flowers dataset using SVM and find out the flower type depending on the given input
   data like sepal length, sepal width, petal length and petal width Find accuracy of all SVM kernels.

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
data = pd.read_csv("iris.csv")
X = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = data["Species"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
kernels = ['linear', 'poly', 'rbf', 'sigmoid']

for kernel in kernels:

  model = SVC(kernel=kernel)

  model.fit(X_train, y_train)

  y_pred = model.predict(X_test)

  accuracy = accuracy_score(y_test, y_pred)
  print(f"Kernel: {kernel}")
  print(f"Accuracy: {accuracy:.4f}")
  print(classification_report(y_test, y_pred))
model=SVC(kernel="rbf")
model.fit(X_train,y_train)
flower_type=model.predict(scaler.fit_transform([[5.1, 3.5, 1.4, 0.2]]))
print(f"The predicted flower type is: {flower_type}")
7) Create KNN model on Indian diabetes patient’s database and predict whether a new patient is diabetic
   (1) or not (0). Find optimal value of K.

   import pandas as pd
   import numpy as np
   from sklearn.model_selection import train_test_split, cross_val_score
   from sklearn.neighbors import KNeighborsClassifier
   from sklearn.preprocessing import StandardScaler
   from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
   import matplotlib.pyplot as plt
   data = pd.read_csv('diabetes.csv')
   print("Dataset Sample:")
   print(data.head())
   X = data.drop(columns=['Outcome']) # Features
   y = data['Outcome']            # Target variable (1 = Diabetic, 0 = Non-diabetic)
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   scaler = StandardScaler()
   X_train = scaler.fit_transform(X_train)
   X_test = scaler.transform(X_test)
   k_range = range(1, 26)
   cv_scores = [] # Store accuracy for each value of k

   for k in k_range:
      knn = KNeighborsClassifier(n_neighbors=k)
      scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
      cv_scores.append(scores.mean())
   plt.plot(k_range, cv_scores)
   plt.xlabel('Number of Neighbors (k)')
   plt.ylabel('Cross-Validated Accuracy')
   plt.title('Finding Optimal k')
   plt.show()
   optimal_k = k_range[np.argmax(cv_scores)]
   print(f"The optimal number of neighbors is {optimal_k}”)

   knn_optimal = KNeighborsClassifier(n_neighbors=optimal_k)
   knn_optimal.fit(X_train, y_train)
   y_pred = knn_optimal.predict(X_test)
   accuracy = accuracy_score(y_test, y_pred)
   print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
   print("\nClassification Report:")
   print(classification_report(y_test, y_pred))
   print("\nConfusion Matrix:")
   print(confusion_matrix(y_test, y_pred))
8) Take iris flower dataset and reduce 4D data to 2D data using PCA. Then train the model and predict
   new flower with given measurements.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data=pd.read_csv("iris.csv")
x = data[["SepalLengthCm","SepalWidthCm","PetalLengthCm","PetalWidthCm"]]
y = data["Species"]

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(x)

X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Model Accuracy: {accuracy * 100:.2f}%')

new_flower = [[5.9, 3.0, 5.1, 1.8]]

new_flower_reduced = pca.transform(new_flower)

predicted_class = knn.predict(new_flower_reduced)
print(predicted_class)
9) Use K-means clustering model and classify the employees into various income groups or clusters.
   Preprocess data if require (i.e. drop missing or null values). Use elbow method and Silhouette Score
   to find value of k.


import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('employee_data.csv')
df.dropna(inplace=True)
data = df[['Age', 'Experience', 'Income']]
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)
inertia = []
for k in range(1, 11):
   kmeans = KMeans(n_clusters=k, random_state=42)
   kmeans.fit(scaled_data)
   inertia.append(kmeans.inertia_)
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia (Sum of Squared Distances)')
plt.title('Elbow Method for Optimal k')
plt.show()
silhouette_scores = []
for k in range(2, 11): # Silhouette score is not defined for k=1
   kmeans = KMeans(n_clusters=k, random_state=42)
   cluster_labels = kmeans.fit_predict(scaled_data)
   silhouette_scores.append(silhouette_score(scaled_data, cluster_labels))
plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker='o', color='r')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score for Different k Values')
plt.show()
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(scaled_data)
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='Income', y='Experience', hue='Cluster', palette='Set1', s=100)
plt.title('Employee Clusters Based on Income and Experience')
plt.show()
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:\n", cluster_centers)
10) The data set refers to clients of a wholesale distributor. It includes the annual spending in monetary
    units on diverse product categories. Using data Wholesale customer dataset compute agglomerative
    clustering     to     find    out     annual    spending        clients   in    the    same    region.
    https://archive.ics.uci.edu/dataset/292/wholesale+customers

   import pandas as pd
   from sklearn.preprocessing import StandardScaler
   from sklearn.cluster import AgglomerativeClustering
   import scipy.cluster.hierarchy as sch
   import matplotlib.pyplot as plt

   data = pd.read_csv('wholesale.csv')

   X = data.drop(['Region', 'Channel'], axis=1)

   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   agg_clustering = AgglomerativeClustering(n_clusters=3, linkage='ward')

   clusters = agg_clustering.fit_predict(X_scaled)

   data['Cluster'] = clusters
   print(data)

   dendrogram = sch.dendrogram(sch.linkage(X_scaled, method='ward'))
   plt.title('Dendrogram')
   plt.xlabel('Customers')
   plt.ylabel('Euclidean Distances')
   plt.show()

   print(data['Cluster'].value_counts())

   cluster_summary = data.groupby('Cluster').mean()
   print(cluster_summary)
11) Use Apriori algorithm on groceries dataset to find which items are brought together. Use minimum
    support =0.25

   import pandas as pd
   from mlxtend.preprocessing import TransactionEncoder
   from mlxtend.frequent_patterns import apriori, association_rules
   groceries = [
      ['milk', 'bread', 'eggs'],
      ['milk', 'bread'],
      ['milk', 'butter', 'bread'],
      ['bread', 'eggs'],
      ['butter', 'eggs']
   ]

   te = TransactionEncoder()
   te_ary = te.fit(groceries).transform(groceries)
   df = pd.DataFrame(te_ary, columns=te.columns_)

   print("Binary Matrix of Transactions:")
   print(df)


   frequent_itemsets = apriori(df, min_support=0.25, use_colnames=True)

   print("\nFrequent Itemsets:")
   print(frequent_itemsets)

   rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

   print("\nAssociation Rules:")
   print(rules[['antecedents', 'consequents', 'support', 'confidence']])
12) Create a two layered neural network with relu and sigmoid activation function.


import numpy as np

np.random.seed(42)

X = np.array([[0.1, 0.2, 0.3],

         [0.4, 0.5, 0.6]])

n_input = 2

n_hidden = 4

n_output = 1

W1 = np.random.randn(n_hidden, n_input) * 0.01

b1 = np.zeros((n_hidden, 1))

W2 = np.random.randn(n_output, n_hidden) * 0.01

b2 = np.zeros((n_output, 1))

Z1 = np.dot(W1, X) + b1 # Linear transformation

A1 = np.maximum(0, Z1) # ReLU activation

Z2 = np.dot(W2, A1) + b2 # Linear transformation

A2 = 1 / (1 + np.exp(-Z2)) # Sigmoid activation

print("Network Output:\n", A2)
13) Create an ANN and train it on house price dataset classify the house price is above average or below
    average.

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
data = fetch_california_housing()
X = data.data # Features
y = data.target # Target (house prices)
average_price = np.mean(y)
y_binary = np.where(y > average_price, 1, 0)
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = Sequential()
model.add(Dense(16, input_shape=(X_train.shape[1],), activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
y_pred = model.predict(X_test) > 0.5
print("\nSample Predictions:")
for i in range(5):
  print(f"Predicted: {'Above Average' if y_pred[i] else 'Below Average'}, Actual: {'Above Average' if
y_test[i] else 'Below Average'}")
14) Create a CNN model and train it on mnist handwritten digit dataset. Using model find out the digit
    written by a hand in a given image. Import mnist dataset from tensorflow.keras.datasets

import numpy as np

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import mnist

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(-1, 28, 28, 1).astype('float32') / 255.0

X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

y_train = to_categorical(y_train, 10)

y_test = to_categorical(y_test, 10)

model = Sequential([Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)),

  MaxPooling2D(pool_size=(2,2)), Conv2D(64, kernel_size=(3,3), activation='relu'),

  MaxPooling2D(pool_size=(2,2)),Flatten(), Dense(128, activation='relu'),

  Dense(10, activation='softmax')])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=128)

loss, accuracy = model.evaluate(X_test, y_test)

print(f"Test Accuracy: {accuracy * 100:.2f}%")

sample_image = X_test[0].reshape(1,28,28,1) # Reshape to match input shape

predicted_class = np.argmax(model.predict(sample_image))

print(f"Predicted digit: {predicted_class}")
15) Write a python program to find all null values in a given data set and remove them.

import pandas as pd
import numpy as np

data=pd.read_csv("ass2_data.csv")
print(data)

print(data.isnull())

print(data.notnull())

data1=data.dropna(axis=0,how="any")
print(data1)

data["m1"]=data["m1"].replace(np.NaN,data["m1"].mean())
data["m2"]=data["m2"].replace(np.NaN,data["m2"].mean())
data["m3"]=data["m3"].replace(np.NaN,data["m3"].mean())
print(data)
16) Write a python program the Categorical values in numeric format for a given dataset.

import pandas as pd
import numpy as np

data=pd.read_csv("ass3_data.csv")
print(data)

x=data.iloc[:,0:1].values
print(x)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
x1=le.fit_transform(x)
print(x1)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()
xn=ohe.fit_transform(x).toarray()
print(xn)

