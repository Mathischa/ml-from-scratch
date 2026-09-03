# Machine Learning Labs

Implémentations from scratch (NumPy) réalisées dans le cadre du module Machine Learning du Master Finance & Ingénierie Quantitative à l'ECE Paris.

## Contenu

- **tp1-regression-lineaire** — Régression linéaire par solution analytique (closed-form) et par descente de gradient stochastique (SGD).
- **tp2-reseau-neurones** — Réseau de neurones feedforward (FFNN) avec rétropropagation, entraîné sur un problème de classification.
- **tp3-kmeans** — Clustering K-means (assignation et mise à jour des centroïdes) implémenté sans bibliothèque de ML.

## Utilisation

```bash
pip install -r requirements.txt
python tp1-regression-lineaire/closed_form_solution.py
python tp1-regression-lineaire/gradient_descent_stochastique.py
python tp2-reseau-neurones/ffnn.py
python tp3-kmeans/kmeans.py
```

Chaque script charge son propre jeu de données (fichier `.txt` du même dossier) et affiche les résultats via matplotlib.
