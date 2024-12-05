#%%
# Fetch data from binance
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from utils.fetch_data import fetch_binance_data

#data = fetch_binance_data("BTCUSDT", "1d", 1640995200000, 1672444800000)
data = fetch_binance_data("BTCUSDT", "5m", "2024-01-01", "2024-3-31")

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
from utils.plot_utils import plot_predictions
# Prédiction
y_pred = model.predict(X_test)
print("MSE:",y_pred)
plot_predictions(y_test, y_pred, title="BTC/USDT Predictions")

#%%
from model.simple_model import CryptoPriceModel
from sklearn.preprocessing import MinMaxScaler
# Configuration
window_size = 20
input_features = 3  # Close, Sin_Day, Cos_Day

# Mise à l'échelle des caractéristiques
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(df[["Close", "Sin_Day", "Cos_Day"]].values)

# Création des séquences
X, y = create_sequences(df["Close"].values, scaled_features, window_size)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")


# Création du modèle
crypto_model = CryptoPriceModel(window_size, input_features)

history = crypto_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32
)

loss, mae = crypto_model.evaluate(X_test, y_test)
print(f"Test Loss: {loss:.4f}, Test MAE: {mae:.4f}")

import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title("Loss Over Epochs")
plt.show()



# Prédiction
# Exemple : Prédire la suite d'une séquence
latest_sequence = X_test[-1:]  # Prenez la dernière séquence de test
predicted_value = crypto_model.predict(latest_sequence)
print(f"Raw Predictions: {y_pred[:10]}")

y_pred = crypto_model.predict(X_test)
print(f"Raw Predictions: {y_pred[:10]}")

print(f"Predicted next value: {predicted_value}")


def predict_future(model, seed_sequence, n_steps, scaler=None):
    """
    Génère des prédictions pour plusieurs pas de temps dans le futur.

    Parameters:
    - model: le modèle entraîné.
    - seed_sequence: la séquence initiale pour commencer les prédictions.
    - n_steps: nombre de valeurs futures à prédire.
    - scaler: le scaler utilisé pour normaliser les données (si applicable).

    Returns:
    - Une liste de prédictions futures.
    """
    predictions = []
    current_sequence = seed_sequence.copy()

    for _ in range(n_steps):
        # Prédire la prochaine valeur
        next_value = model.predict(current_sequence)[0][0]
        predictions.append(next_value)

        # Mettre à jour la séquence : faites glisser les données et ajoutez la prédiction
        next_sequence = np.roll(current_sequence, -1, axis=1)
        next_sequence[0, -1] = next_value
        current_sequence = next_sequence

    if scaler:
        # Créer une matrice fictive pour appliquer inverse_transform correctement
        scaled_predictions = np.zeros((len(predictions), scaler.n_features_in_))
        scaled_predictions[:, 0] = predictions  # Colonne 0 correspond à `Close`
        predictions = scaler.inverse_transform(scaled_predictions)[:, 0]  # Récupérer uniquement `Close`

    return predictions


# Exemple : prédire 10 pas de temps dans le futur
future_predictions = predict_future(
    crypto_model,
    latest_sequence,
    n_steps=10,
    scaler=scaler  # si les données sont scalées
)

print(f"Future predictions: {future_predictions}")

plt.figure(figsize=(12, 6))
plt.plot(range(len(y_test)), y_test, label="Real Prices")
plt.plot(range(len(y_test), len(y_test) + len(future_predictions)), future_predictions, label="Predicted Future Prices", linestyle="--")
plt.legend()
plt.title("BTC/USDT Predictions")
plt.show()
