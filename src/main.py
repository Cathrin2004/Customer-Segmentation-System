import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Show first 5 rows
print(df.head())

# Rename column
df.rename(columns={"Genre": "Gender"}, inplace=True)

# Check missing values
print(df.isnull().sum())

# Feature selection
X = df[["Annual Income (k$)", "Spending Score (1-100)"]]

# Feature scaling
scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# Elbow method
clustering_error = []

for i in range(1, 11):

    kmeans = KMeans(n_clusters=i, random_state=42)

    kmeans.fit(X_scaled)

    clustering_error.append(kmeans.inertia_)

# Plot elbow graph
plt.figure(figsize=(8, 5))

plt.plot(range(1, 11), clustering_error, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("Clustering Error")

plt.savefig("outputs/elbow_method.png")

plt.show()

# Apply K-Means
kmeans = KMeans(n_clusters=5, random_state=42)

y_kmeans = kmeans.fit_predict(X_scaled)

# Add cluster column
df["Cluster"] = y_kmeans

# Visualize clusters
plt.figure(figsize=(8, 6))

plt.scatter(
    X_scaled[:, 0],
    X_scaled[:, 1],
    c=y_kmeans,
    cmap='viridis'
)

plt.title("Customer Segments")

plt.xlabel("Annual Income (scaled)")

plt.ylabel("Spending Score (scaled)")

plt.savefig("outputs/customer_segments.png")

plt.show()

# Cluster summary
cluster_summary = df.groupby("Cluster")[
    ["Annual Income (k$)", "Spending Score (1-100)"]
].mean()

print(cluster_summary)