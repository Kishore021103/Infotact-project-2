import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")

# 1. Read the CSV file (it must be in the same folder as this .py file)
df = pd.read_csv("Mall_Customers.csv")

# 2. Show first 5 rows
print("HEAD:")
print(df.head())

# 3. Show info (columns, data types, missing values)
print("\nINFO:")
print(df.info())

# 4. Show basic stats
print("\nDESCRIBE:")
print(df.describe())

plt.figure()
sns.histplot(df["Age"], bins=10, kde=True)
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

plt.figure()
sns.histplot(df["Annual Income (k$)"], bins=10, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Count")
plt.show()

plt.figure()
sns.histplot(df["Spending Score (1-100)"], bins=10, kde=True)
plt.title("Spending Score Distribution")
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Count")
plt.show()

plt.figure()
sns.scatterplot(
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    data=df
)
plt.title("Annual Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# STEP 4: Select features for clustering
X = df[["Age", "Annual Income (k$)", "Spending Score (1-100)"]]
print("\nSelected Features:")
print(X.head())

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled Data (first 5 rows):")
print(X_scaled[:5])

# STEP 5: Find optimal number of clusters (Elbow Method)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia_list = []
K_range = range(1, 11)  # Trying K from 1 to 10

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia_list.append(kmeans.inertia_)

# Plot the Elbow Graph
plt.figure()
plt.plot(K_range, inertia_list, marker='o')
plt.title("Elbow Method - Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (SSE)")
plt.show()

# STEP 6: Apply K-Means with optimal K
best_k = 5
kmeans = KMeans(n_clusters=best_k, random_state=42)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to original dataframe
df["Cluster"] = cluster_labels

print("\nCluster labels added:")
print(df.head())

# STEP 7: Visualize clusters (Income vs Spending Score)
plt.figure()
sns.scatterplot(
    x=df["Annual Income (k$)"],
    y=df["Spending Score (1-100)"],
    hue=df["Cluster"],
    palette="tab10"
)
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()

# STEP 8: Cluster profiling

# 1) How many customers in each cluster?
print("\nNumber of customers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# 2) Average Age, Income, Spending Score per cluster
cluster_profile = df.groupby("Cluster")[["Age",
                                         "Annual Income (k$)",
                                         "Spending Score (1-100)"]].mean()

print("\nCluster profile (means):")
print(cluster_profile)

# 3) Gender distribution in each cluster (optional but useful)
print("\nGender % in each cluster:")
gender_dist = df.groupby("Cluster")["Genre"].value_counts(normalize=True) * 100
print(gender_dist)
