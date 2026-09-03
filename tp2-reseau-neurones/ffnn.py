# Importation des bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Lecture des données
name_file = 'data_ffnn.txt'
columns = ['x1', 'x2', 'y']
data_in = pd.read_csv(name_file, names=columns, sep='\t', skiprows=1)
x1 = data_in['x1'].values
x2 = data_in['x2'].values
y = data_in['y'].astype(int).values

# Matrice d'entrée
X = np.column_stack((x1, x2))

# 1. Tracer les données
plt.scatter(x1, x2, c=y, cmap='viridis', edgecolor='k', s=50)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Données classées')
plt.show()

# Paramètres du réseau de neurones
input_dim = 2 #
hidden_dim = 5  #K # Vous pouvez choisir une valeur différente
output_dim = len(np.unique(y))

# Initialisation des poids
np.random.seed(0)
#dim N+1 * K
V = np.random.randn(input_dim + 1, hidden_dim) * 0.1  # +1 pour le biais
#dim K+1 * J
W = np.random.randn(hidden_dim + 1, output_dim) * 0.1  # +1 pour le biais

# Paramètres d'apprentissage
alpha1 =  0.01
alpha2 = 0.06
Threshold = 1e-2

# Fonctions d'activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Calcul de l'erreur SSE
def calculate_sse(G, Y):
    return 0.5 * np.sum((G - Y) ** 2)

# Encodage one-hot des étiquettes
Y_one_hot = np.zeros((y.size, output_dim))
Y_one_hot[np.arange(y.size), y] = 1

# Fonction de propagation avant (FWP)
def forward_propagation(X, V, W):
    X_barre = np.column_stack((np.ones(X.shape[0]), X))  # Ajouter la colonne de biais
    X_barre_barre = X_barre.dot(V)  # Pré-activation de la première couche
    F = sigmoid(X_barre_barre)  # Activation de la première couche
    F_barre = np.column_stack((np.ones(F.shape[0]), F))  # Ajouter la colonne de biais
    F_barre_barre = F_barre.dot(W)  # Pré-activation de la couche de sortie
    G = sigmoid(F_barre_barre)  # Activation de la sortie
    return X_barre, F, F_barre, G

# Fonction de rétropropagation (BP)
def backpropagation(X_barre, F, F_barre, G, Y_one_hot, V, W, alpha1, alpha2):
    # Calcul des gradients pour W
    delta_G = (G - Y_one_hot) * sigmoid_derivative(G)
    grad_W = F_barre.T.dot(delta_G)

    # Calcul des gradients pour V
    delta_F = delta_G.dot(W[1:].T) * sigmoid_derivative(F)
    grad_V = X_barre.T.dot(delta_F)

    # Mise à jour des poids
    W -= alpha1 * grad_W
    V -= alpha2 * grad_V

    return V, W

# Boucle d'apprentissage
E = float('inf')
Delta_E = float('inf')
itera = 0
error_history = []

while abs(Delta_E) > Threshold:
    # Propagation avant
    X_barre, F, F_barre, G = forward_propagation(X, V, W)

    # Calcul de l'erreur
    E_prev = E
    E = calculate_sse(G, Y_one_hot)
    Delta_E = E_prev - E
    error_history.append(E)

    # Rétropropagation
    V, W = backpropagation(X_barre, F, F_barre, G, Y_one_hot, V, W, alpha1, alpha2)

    print(f"Itération {itera}, Erreur : {E}, Delta Erreur : {Delta_E}")

    # Vérification de la condition d'arrêt
    if abs(Delta_E) <= Threshold:
        break

    # Incrémentation de l'itération
    itera += 1

print("Apprentissage terminé.")



# 3. Tracer la réduction de l'erreur
plt.plot(error_history)
plt.xlabel('Itérations')
plt.ylabel('Erreur SSE')
plt.title('Réduction de l\'erreur au cours des itérations')
plt.show()

# 4. Affichage des poids optimaux
print("Poids optimaux V (couche cachée) :\n", V)
print("Poids optimaux W (couche de sortie) :\n", W)

# 5. Comparer les valeurs de sortie
print("Valeurs de sortie prédites (G) :\n", np.argmax(G, axis=1))
print("Valeurs réelles (y) :\n", y)

# 6. Tester avec les nouvelles données
test_data = np.array([[0, 0], [2, 2], [4, 4], [4.5, 1.5]])
_, _, _, G_test = forward_propagation(test_data, V, W)
print("Prédictions pour les nouvelles données :\n", np.argmax(G_test, axis=1))

# 7. Tracer les résultats de la classification
plt.scatter(x1, x2, c=y, cmap='viridis', edgecolor='k', s=50, label='Données d\'entraînement')
plt.scatter(test_data[:, 0], test_data[:, 1], c=np.argmax(G_test, axis=1), cmap='cool', marker='x', s=100, label='Nouvelles données')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Classification des données')
plt.legend()
plt.show()
