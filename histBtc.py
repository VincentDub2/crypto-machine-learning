#%%
# Fetch data from binance
import matplotlib.pyplot as plt
import numpy as np
import requests
from utils.date_utils import date_to_timestamp
from utils.plot_utils import plot_predictions

url = "https://api.binance.com/api/v3/klines"

params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": date_to_timestamp("2024-01-01"),  # Convertir la date de début
    "endTime": date_to_timestamp("2024-12-31")     # Convertir la date de fin
}

response = requests.get(url, params=params)
data = response.json()

print(data)

# Conversion en DataFrame pour analyse
import pandas as pd
df = pd.DataFrame(data, columns=[
    "Open time", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
])

print(df.head())

# Save to a CSV file
csv_file_name = "btc_usdt_daily_prices.csv"
df.to_csv(csv_file_name, index=False)
print(f"Data has been saved to {csv_file_name}")


#%%
# Display data
# Création du DataFrame

print("Création du DataFrame")

df = pd.DataFrame(data, columns=[
    "OpenTime", "Open", "High", "Low", "Close", "Volume",
    "Close time", "Quote asset volume", "Number of trades",
    "Taker buy base asset volume", "Taker buy quote asset volume", "Ignore"
])

print('Affichage des premières lignes')

# Affichage des premières lignes
df.columns = df.columns.str.strip()
print(df.columns)

print("conversion de Open time au format datetime")
# Conversion de la colonne "Open time" en format date lisible
df["OpenTime"] = pd.to_datetime(df["OpenTime"], unit="ms")

df["Open"] = pd.to_numeric(df["Open"])
df["Close"] = pd.to_numeric(df["Close"])
df["DayOfYear"] = df["OpenTime"].dt.dayofyear
df["Sin_Day"] = np.sin(2 * np.pi * df["DayOfYear"] / 365.25)
df["Cos_Day"] = np.cos(2 * np.pi * df["DayOfYear"] / 365.25)


# Vérification de la conversion
print(df.head())


# Création du graphique
ax = df.plot(
    x="OpenTime",
    y=["Open", "Close"],
    figsize=(12, 6),
    title="BTC/USDT Daily Prices",
)

print("Ajout d'une grille et légende")

# Ajout d'une grille et légende
ax.grid()
ax.legend(["Open Price", "Close Price"])

# Affichage du graphique
plt.show()

#%%
# Préparation des données
import numpy as np
def create_sequences(data, features, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        seq_features = features[i:i + window_size]
        seq_target = data[i + window_size]
        X.append(seq_features)
        y.append(seq_target)
    return np.array(X), np.array(y)

# Features utilisées : Close, Sin_Day, Cos_Day
window_size = 120
features = df[["Close", "Sin_Day", "Cos_Day"]].values
X, y = create_sequences(df["Close"].values, features, window_size)

# Séparer en données d'entraînement et de test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#%%
# Création du modèle

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Mise à plat des séquences pour la régression linéaire
X_train_flat = X[:train_size].reshape(train_size, -1)
X_test_flat = X[train_size:].reshape(len(X) - train_size, -1)

# Entraîner et prédire
model = LinearRegression()
model.fit(X_train_flat, y_train)

y_pred = model.predict(X_test_flat)
print("MSE:", mean_squared_error(y_test, y_pred))


#%%
plt.figure(figsize=(12, 6))
plt.plot(y_test, label="Real Prices")
plt.plot(y_pred, label="Predicted Prices")
plt.legend()
plt.title("BTC/USDT Predictions")
plt.show()


#%%
# Création du modèle
# Reshape pour LSTM : (samples, timesteps, features)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Création du modèle
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(window_size, 3)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=20, batch_size=32)


#%%
from utils.plot_utils import plot_predictions
# Prédiction
y_pred = model.predict(X_test)
print("MSE:",y_pred)
plot_predictions(y_test, y_pred, title="BTC/USDT Predictions")
