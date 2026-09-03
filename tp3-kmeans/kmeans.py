import numpy as np
import matplotlib.pyplot as plt

# Chargement des données et paramètres
X = np.loadtxt("data_kmeans.txt")
I, N = X.shape  # I est le nombre de points, N est le nombre de variables d'entrée
K = 3  # Nombre de clusters

np.random.seed(1)
mean_data = np.mean(X, axis=0)
std_data = np.std(X, axis=0)

# Génération de données de test
test_data = np.random.normal(mean_data, std_data, (10, N))  # 10 points avec même moyenne et écart-type

# Initialisation aléatoire des centroides
np.random.seed(0)
mu = X[np.random.choice(I, K, replace=False)]  # Choisir K points initiaux pour les centroides

# Fonctions pour K-means
def compute_clusters(X, mu):
    """assigne chaque point de données à un cluster en fonction de sa distance par rapport aux centroides. (minimiser la distance au carré)."""
    distance_d = np.sum((X[:, np.newaxis, :] - mu[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(distance_d, axis=1)

def update_centroids(X, clusters, K): 
    """Calcule les nouveaux centroides comme la moyenne des points dans chaque cluster."""
    new_mu = np.array([X[clusters == k].mean(axis=0) for k in range(K)])
    return new_mu

def k_means(X, mu, K, max_iters=1000):
    """Applique l'algorithme K-means."""
    for _ in range(max_iters):
        clusters = compute_clusters(X, mu)  # Étape d'assignation
        new_mu = update_centroids(X, clusters, K)  # Étape de mise à jour des centroides
        if np.allclose(mu, new_mu):  # Convergence
            break
        mu = new_mu
    return clusters, mu

# Appliquer K-means sur les données d'entraînement
clusters, mu = k_means(X, mu, K)

# Calculer les clusters pour les données de test
test_clusters = compute_clusters(test_data, mu)

# Affichage des résultats
plt.figure(figsize=(8, 6))

# Graphique pour les données d'entraînement
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', label="Données d'entraînement")
plt.scatter(mu[:, 0], mu[:, 1], c='red', marker='x', s=100, label="Centroids")

# Graphique pour les données de test, en utilisant les mêmes couleurs que les clusters d'entraînement
plt.scatter(test_data[:, 0], test_data[:, 1], c=test_clusters, cmap='viridis', marker='D', s=70, edgecolor='black', label="Données de test")

# Légendes et affichage
plt.title("KMeans : Résultats sur données d'entraînement et de test")
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend()
plt.show()
