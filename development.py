from demo_code import *

symbol = 'SPY'
start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2015, 12, 31)
sp500 = web.DataReader(symbol, "yahoo", start, end)

# Plot last year of price "candles"
plot_candlesticks(sp500, datetime.datetime(2015, 1, 1))

# Carry out K-Means clustering with five clusters on the
# three-dimensional data H/O, L/O and C/O
sp500_norm = get_open_normalised_prices(symbol, start, end)
k = 5
km = KMeans(n_clusters=k, random_state=42)
km.fit(sp500_norm)
labels = km.labels_
sp500["Cluster"] = labels

# Plot the 3D normalised candles using H/O, L/O, C/O
plot_3d_normalised_candles(sp500_norm, labels)

# Plot the full OHLC candles re-ordered
# into their respective clusters
plot_cluster_ordered_candles(sp500)

# Create and output the cluster follow-on matrix
create_follow_cluster_matrix(sp500, k)


# get average values of candles in clusters
sp500_norm['Cluster'] = labels
sp500_norm.groupby('Cluster').mean()
