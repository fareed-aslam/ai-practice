import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

data = {
    "student_id": range(1, 21),
    "GPA": [2.5, 3.0, 3.5, 3.8, 2.2, 2.9, 3.2, 3.6, 2.7, 3.1,
            3.9, 2.4, 3.3, 3.7, 2.8, 3.0, 3.4, 2.6, 3.8, 3.2],
    "study_hours": [10, 15, 20, 25, 8, 14, 18, 22, 12, 16,
                    28, 9, 19, 24, 13, 15, 21, 11, 26, 17],
    "attendance_rate": [60, 70, 80, 90, 55, 65, 75, 85, 68, 72,
                        95, 58, 78, 88, 66, 70, 82, 62, 92, 76]
}

df = pd.DataFrame(data)

features = df[['GPA', 'study_hours', 'attendance_rate']]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

inertia = []

K_range = range(2, 7)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(K_range, inertia, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

optimal_k = 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_features)

plt.figure()
plt.scatter(df['study_hours'], df['GPA'], c=df['cluster'])
plt.xlabel("Study Hours")
plt.ylabel("GPA")
plt.title("Student Clusters")
plt.show()

print(df)
