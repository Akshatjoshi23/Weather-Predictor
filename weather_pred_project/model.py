import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle

df = pd.read_csv("weather.csv")

X = df[['Temperature', 'Humidity', 'Windy']]
y = df['Weather']

model = DecisionTreeClassifier()
model.fit(X, y)

with open("weather_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved.")
