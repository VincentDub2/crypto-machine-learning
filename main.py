#%%
# Import libraries
import pandas as pd
import matplotlib.pyplot as plt

#%%
# Fetch data from binance

import requests

url = "https://api.binance.com/api/v3/klines"
params = {
    "symbol": "BTCUSDT",
    "interval": "1d",
    "startTime": 1640995200000,  # Timestamp en millisecondes
    "endTime": 1672444800000,    # Timestamp en millisecondes
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
import matplotlib.pyplot as plt

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
# Display data

print(df.head())

# Convert "Open time" to a readable date format
#df["Open time"] = pd.to_datetime(df["Open time"], unit="ms")

# Ensure numerical columns are converted to numeric types
df["Open"] = pd.to_numeric(df["Open"])
df["Close"] = pd.to_numeric(df["Close"])

# Plot using pandas' built-in plotting functionality
ax = df.plot(
    x="Open time",
    y=["Open", "Close"],
    figsize=(12, 6),
    title="BTC/USDT Daily Prices",
    ylabel="Price (USDT)",
    xlabel="Date"
)

# Customize grid and legend
ax.grid()
ax.legend(["Open Price", "Close Price"])

# Show the plot
plt.title('BTC/USDT Daily Prices')
plt.show()
