import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

used_cars = pd.read_csv(r"C:\Users\Anshika\Downloads\used_cars.csv")
print(used_cars.head())
print(used_cars.isnull().sum())
print(used_cars.info())
print(used_cars.describe())

# Vehicle count by manufacturer
figsize=(15,5)
brands = used_cars.sort_values(by="manufacturer", ascending=False)
sns.catplot(y="manufacturer", data=brands, kind="count", height=7, aspect=1.5, palette="plasma")
plt.title("No.of Vehicles sold", fontdict={'size': 20})
plt.xlabel("Vehicle count", fontdict={'size': 15})
plt.ylabel("Car Brands", fontdict={'size': 15})
plt.xticks(rotation=90)
plt.show()

# Avg selling price according to manufacturer
used_cars.groupby("manufacturer")["price"].mean().plot(kind="bar", figsize=(15,5), color="palevioletred")
plt.title("Avg. Selling Price", fontdict={'size': 20})
plt.xlabel("Type Of Vehicle", fontdict={'size': 15})
plt.ylabel("Selling Price", fontdict={'size': 15})
plt.xticks(rotation=90)
plt.show()

# Sales of vehicles by year
fig, ax = plt.subplots(figsize=(15,5))
sns.histplot(used_cars["year"], color="#33cc33", kde=True, ax=ax)
ax.set_title('Distribution of vehicles based on Year', fontsize=15)
plt.ylabel("Count", fontsize=15)
plt.xlabel("Years", fontsize=15)
plt.show()

# Mileage by brand with respect to transmission
fig, ax = plt.subplots(figsize=(15,5))
sns.lineplot(data=used_cars, x='manufacturer', y='mileage', hue='transmission')
plt.xticks(rotation=90)
ax.set_title('Manual and Automatic Car Mileage', fontsize=15)
plt.show()

# Scatter plot - engine vs mpg
fig, ax = plt.subplots(figsize=(15,5))
sns.scatterplot(data=used_cars, x="engine", y="mpg")
plt.title("Relationship between Engine and Mileage", fontdict={'size': 20})
plt.xticks(rotation=90)
plt.show()

# Regression - Predicting price based on mileage, year, mpg
X = used_cars[["mileage", "year", "mpg"]]
y = used_cars["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse:.2f}')
