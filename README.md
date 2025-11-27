# K-Means Improved Project

This package contains an enhanced NumPy-only implementation of K-Means clustering with modern best-practices:
- kmeans++ initialization (to improve centroid seeding)
- multiple restarts (n_init) to avoid poor local minima
- inertia computation and convergence tolerance
- programmatic selection of K via Silhouette and Elbow heuristics
- visualizations and analysis

Files included:
- kmeans_improved.py : NumPy KMeans implementation
- X.npy, y_true.npy : synthetic dataset (ground truth 4 clusters)
- silhouette_scores.png, elbow_inertia.png, final_clusters.png : plots
- analysis.txt, optimal_k.txt : textual outputs and choice rationale
- requirements.txt : minimal python deps

Usage example (Python):
```python
import numpy as np
from kmeans_improved import KMeansScratch
X = np.load('X.npy')
model = KMeansScratch(n_clusters=4, random_state=42, n_init=8)
model.fit(X)
print(model.inertia_, model.cluster_centers_)
```

Generated with deterministic randomness for reproducibility.
