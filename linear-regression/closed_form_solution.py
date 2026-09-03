import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Chemin du fichier contenant les données
name_file = './data_lab1.txt'

# Définition des noms des colonnes à utiliser pour les données lues
columns = ['x', 'y']

# Lecture du fichier texte en utilisant pandas
data_in = pd.read_csv(name_file, names=columns, sep=' ')

# Conversion des colonnes 'x' et 'y' du DataFrame en tableaux numpy
x = np.asarray(data_in['x'])  # Valeurs d'entrée (features)
y = np.asarray(data_in['y'])  # Valeurs de sortie (target)

# Nombre d'exemples
m = len(x)

# Division des données
train_size = int(0.7 * m)  # 70% des données 
x_train = x[:train_size]  # x_train est le sous-ensemble des 70% premiers exemples
y_train = y[:train_size]  # y_train est le sous-ensemble des valeurs cibles associées à x_train
x_test = x[train_size:]  # x_test est le sous-ensemble des 30% derniers exemples pour le test
y_test = y[train_size:]  # y_test est le sous-ensemble des valeurs cibles associées à x_test

# ------------------------- Application de la Closed-Form Solution -------------------------

# Ajouter une colonne de 1 à x_train pour représenter l'ordonnée à l'origine (theta_0)
X_train = np.c_[np.ones((train_size, 1)), x_train]  # Matrice X avec une colonne de 1 et x_train

# Calcul de la solution analytique (Closed-Form Solution) pour theta
theta_optimal = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

# Extraire les valeurs optimales de theta_0 et theta_1
theta_0_opt = theta_optimal[0]  # Ordonnée à l'origine (intercept)
theta_1_opt = theta_optimal[1]  # Pente de la droite (slope)

# Affichage des valeurs optimales
print(f"Valeur optimale de theta_0 (CFS - intercept): {theta_0_opt}")
print(f"Valeur optimale de theta_1 (CFS - slope): {theta_1_opt}")

# ------------------------- Prédictions sur l'ensemble de test -------------------------

# Ajouter une colonne de 1 à x_test pour la prédiction (en incluant theta_0)
X_test = np.c_[np.ones((len(x_test), 1)), x_test]  # Matrice X_test avec une colonne de 1 et x_test

# Prédictions pour l'ensemble de test
y_test_pred = X_test @ theta_optimal

# Calcul de la SSE pour l'ensemble de test
sse_test = (1/2) * np.sum((y_test_pred - y_test)**2)
print(f"SSE sur l'ensemble de test (CFS): {sse_test}")

# ------------------------- Tracé des résultats -------------------------

plt.figure()

# Tracé des données d'origine
plt.scatter(x, y, color='red', label='Données originales')

# Tracé de la droite de régression pour l'ensemble d'entraînement
y_train_pred = theta_0_opt + theta_1_opt * x_train
plt.plot(x_train, y_train_pred, 'b-', label='Régression - Entraînement (CFS)')

# Tracé de la droite de régression pour l'ensemble de test
y_test_pred = theta_0_opt + theta_1_opt * x_test
plt.plot(x_test, y_test_pred, 'g-', label='Régression - Test (CFS)')

# Configuration du graphique
plt.xlabel('x')
plt.ylabel('y')
plt.title('Régressions linéaires pour Entraînement et Test (CFS)')
plt.legend()
plt.grid(True)
plt.show()