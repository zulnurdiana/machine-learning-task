import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Read data from CSV file
df = pd.read_csv('../dataset/Weather.csv').head(10)

# Drop the 'date' column
df = df.drop('date', axis=1)

# Select the desired columns for clustering
X = df[['precipitation', 'temp_max', 'temp_min', 'wind']]
X = StandardScaler().fit_transform(X)

# Initialize DBSCAN
dbscan = DBSCAN(eps=2, min_samples=4)

# Fit the model to the data
model = dbscan.fit(X)

# Get the cluster labels
labels = model.labels_

# Identify core samples
core_samples_mask = np.zeros_like(labels, dtype=bool)
core_samples_mask[model.core_sample_indices_] = True

# Plot the data and clusters using the first two features for visualization
plt.scatter(X[core_samples_mask, 0], X[core_samples_mask, 1],
            c=labels[core_samples_mask], cmap='viridis', marker='o', s=100, edgecolor='k')

# Plot non-core samples
plt.scatter(X[~core_samples_mask, 0], X[~core_samples_mask, 1],
            c=labels[~core_samples_mask], cmap='viridis', marker='o', s=50, edgecolor='k', alpha=0.6)

# Add a circle for each core sample to show the eps radius
for point in X[core_samples_mask]:
    plt.gca().add_patch(
        Circle(point[:2], 2, fill=False, color='k', linestyle='--', linewidth=1))

# Show the plot
plt.title('DBSCAN Clustering')
plt.xlabel('Feature 1 (Standardized)')
plt.ylabel('Feature 2 (Standardized)')
plt.show()
