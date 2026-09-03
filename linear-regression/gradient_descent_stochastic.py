# Importation des bibliothèques nécessaires pour la manipulation des données et la création de graphiques
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Chemin du fichier contenant les données
name_file = './data_lab1.txt'

# Définition des noms des colonnes à utiliser pour les données lues
columns = ['x', 'y']

# Lecture du fichier texte en utilisant pandas, avec 'names' pour nommer les colonnes et 'sep' pour indiquer le séparateur
data_in = pd.read_csv(name_file, names=columns, sep=' ')

# Conversion des colonnes 'x' et 'y' du DataFrame en tableaux numpy
x = np.asarray(data_in['x'])  # x(i) correspond aux valeurs d'entrée (features)
y = np.asarray(data_in['y'])  # y(i) correspond aux valeurs de sortie (target)

# Création d'une nouvelle figure pour le tracé avec matplotlib
plt.figure()

# Tracé des points 'x' et 'y' en rouge ('ro' signifie points rouges)
plt.plot(x, y, 'ro')

# Ajout d'un label pour l'axe des abscisses (x)
plt.xlabel('x')

# Ajout d'un label pour l'axe des ordonnées (y')
plt.ylabel('y')

# Ajout d'un titre au graphique pour mieux comprendre ce qui est tracé
plt.title('Output value as a function of input data')

# Affichage du graphique final
plt.show()

# ==== Début de la première question ====

# Nombre total de données, m = nombre d'exemples
m = len(x)

# Indice pour la séparation 70/30 pour diviser les données en ensemble d'entraînement et de test
train_size = int(0.7 * m)  # 70% des données pour l'entraînement

# Division des données
x_train = x[:train_size]  # x_train est le sous-ensemble des 70% premiers exemples
y_train = y[:train_size]  # y_train est le sous-ensemble des valeurs cibles associées à x_train
x_test = x[train_size:]  # x_test est le sous-ensemble des 30% derniers exemples pour le test
y_test = y[train_size:]  # y_test est le sous-ensemble des valeurs cibles associées à x_test

# Affichage pour vérifier les tailles
print(f"Ensemble d'entraînement : {len(x_train)} exemples")
print(f"Ensemble de test : {len(x_test)} exemples")

# ==== Début de la deuxième question ====

# Initialisation du paramètre pour la descente de gradient stochastique (SGD)
theta_1 = 0  # Paramètre theta_1 (pente de la droite)
alpha = 0.01  # Taux d'apprentissage (learning rate)
itera = 0  # Compteur d'itérations
threshold = 1e-2  # Critère de convergence
max_iterations = 1000  # Nombre maximal d'itérations pour éviter une boucle infinie
m_train = len(x_train)  # Nombre d'exemples pour l'ensemble d'entraînement

# Fonction pour calculer la SSE (Sum of Squared Errors)
def compute_sse(x, y, theta_1):
    m = len(x)
    y_pred = theta_1 * x
    sse = (1/2) * np.sum((y_pred - y)**2)  # La formule de SSE avec le facteur 1/2
    return sse

# Initialisation de la première SSE pour le critère de convergence
E_prev = compute_sse(x_train, y_train, theta_1)

# Boucle de descente de gradient stochastique (SGD)
while True:
    itera += 1  # Incrémentation du compteur d'itérations

    # Sélectionner un exemple aléatoire dans les données d'entraînement selon une loi uniforme
    i = np.random.randint(0, m_train)  # Choisir un i aléatoire dans [0, m_train-1]

    # Calculer la prédiction pour cet exemple particulier
    y_pred = theta_1 * x_train[i]  # h_theta(x(i)) = theta_1 * x(i)
    
    # Calculer l'erreur pour cet exemple
    error = y_pred - y_train[i]  # (h_theta(x(i)) - y(i))

    # Calculer le gradient pour theta_1 pour cet exemple
    gradient_theta_1 = error * x_train[i]  # Dérivée partielle de E par rapport à theta_1 pour l'exemple i
    
    # Mise à jour du paramètre en utilisant le taux d'apprentissage alpha
    theta_1 -= alpha * gradient_theta_1

    # Calculer l'erreur actuelle (SSE) sur l'ensemble de l'entraînement après avoir parcouru tous les exemples
    E_current = compute_sse(x_train, y_train, theta_1)
    
    # Affichage de l'erreur pour chaque itération
    print(f"Iteration {itera}: SSE = {E_current}")

    # Vérifier le critère de convergence (si la variation de l'erreur est inférieure au threshold)
    if abs(E_prev - E_current) < threshold or itera >= max_iterations:
        print("Convergence atteinte.")
        break
    
    # Mettre à jour la valeur de l'erreur précédente
    E_prev = E_current

# Afficher la valeur optimale du paramètre après convergence
print(f"Valeur optimale de theta_1 (slope): {theta_1}")

# === Fin de la deuxième question ===

# ==== Début de la quatrième question ====

# Calcul des prédictions pour l'ensemble de test
y_test_pred = theta_1 * x_test  # Utilisation de la valeur optimale de theta_1

# Calcul de la SSE pour l'ensemble de test
sse_test = compute_sse(x_test, y_test, theta_1)

# Afficher la SSE pour l'ensemble de test
print(f"SSE sur l'ensemble de test : {sse_test}")

# === Fin de la quatrième question ===

# ==== Début de la cinquième question ====

# Tracé des résultats pour l'ensemble d'entraînement et l'ensemble de test

plt.figure()

# Tracé des données d'origine (ensemble complet)
plt.plot(x, y, 'ro', label='Données originales')  # Données originales en rouge

# Tracé de la droite de régression pour l'ensemble d'entraînement
y_train_pred = theta_1 * x_train
plt.plot(x_train, y_train_pred, 'b-', label='Régression - Entraînement')  # Régression pour l'entraînement en bleu

# Tracé de la droite de régression pour l'ensemble de test
y_test_pred = theta_1 * x_test
plt.plot(x_test, y_test_pred, 'g-', label='Régression - Test')  # Régression pour le test en vert

# Configuration du graphique
plt.xlabel('x')
plt.ylabel('y')
plt.title('Régressions linéaires pour Entraînement et Test')
plt.legend()
plt.show()

# === Fin de la cinquième question ===