# ML From Scratch

Implémentation de trois algorithmes de machine learning fondamentaux en Python/NumPy, sans bibliothèque de ML — uniquement l'algèbre linéaire et le calcul différentiel sous-jacents.

## Contenu

### [linear-regression](linear-regression)
Régression linéaire par solution analytique (équation normale) et par descente de gradient stochastique.

### [neural-network](neural-network)
Réseau de neurones feedforward avec rétropropagation, entraîné sur un problème de classification.

### [kmeans-clustering](kmeans-clustering)
Clustering K-means : assignation aux centroïdes et mise à jour itérative jusqu'à convergence.

## Utilisation

```bash
pip install -r requirements.txt
python linear-regression/closed_form_solution.py
python linear-regression/gradient_descent_stochastic.py
python neural-network/feedforward_neural_network.py
python kmeans-clustering/kmeans.py
```

Chaque script charge son propre jeu de données et affiche les résultats via matplotlib.
