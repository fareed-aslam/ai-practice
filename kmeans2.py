import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "customer_id": range(1, 11),
    "age": [22, 25, 47, 52, 46, 56, 23, 40, 60, 48],
    "income": [15, 16, 40, 42, 38, 45, 14, 35, 50, 41],
    "spending_score": [39, 81, 6, 77, 40, 76, 94, 3, 72, 14]
}

df = pd.DataFrame(data)

X = df.drop(columns=["customer_id"])

kmeans1 = KMeans(n_clusters=3, random_state=42)
df["cluster_no_scaling"] = kmeans1.fit_predict(X)

scaler = StandardScaler()
X_scaled = X.copy()
cols = ["income", "spending_score"]
X_scaled[cols] = scaler.fit_transform(X_scaled[cols])

kmeans2 = KMeans(n_clusters=3, random_state=42)
df["cluster_with_scaling"] = kmeans2.fit_predict(X_scaled)

print(df)
