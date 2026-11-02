# California Housing, prédiction du prix des logements

Chapitre 2 de *Hands-On Machine Learning* (Aurélien Géron), le projet « de bout en bout ».

Le but : prédire le prix médian des logements d'un district de Californie (recensement de 1990) à partir du revenu médian, de la population, de la position... C'est de la régression supervisée, je mesure l'erreur avec la RMSE.

## Les fichiers

- `california_housing_complet.ipynb` : tout le projet. À chaque étape j'ai noté pourquoi je fais ça, la méthode choisie, les alternatives et pourquoi je ne les ai pas prises.
- `datasets/` : téléchargé par le notebook, pas dans git.
- `models/` : le modèle entraîné (`.pkl`, environ 150 Mo), pas dans git.

## Relancer

```bash
cd machine-learning-specialization/Hands_On_Machine_Learning_Projects
pip install -r requirements.txt
jupyter notebook California_Housing_Prices_Prediction/california_housing_complet.ipynb
```

Tout le notebook prend environ 5 minutes, c'est surtout la recherche d'hyperparamètres qui est longue.

## Les étapes

1. Cadrer le problème, choisir la métrique (RMSE).
2. Télécharger et regarder les données : valeurs manquantes (`total_bedrooms`), valeurs plafonnées, distributions asymétriques.
3. Mettre de côté un jeu de test **stratifié** sur le revenu (20 %), **avant** d'explorer.
4. Explorer : carte, corrélations, nouveaux ratios (`bedrooms_ratio`, `rooms_per_house`, `people_per_house`).
5. Tout le prétraitement dans un **pipeline** scikit-learn : imputation, one-hot, log, standardisation, et similarité à 10 clusters géographiques (transformateur perso K-means + RBF). Ça donne 24 features.
6. Comparer les modèles en validation croisée (10 plis).
7. Régler la forêt aléatoire (`GridSearchCV` puis `RandomizedSearchCV`), regarder les importances et les erreurs.
8. Évaluer une seule fois sur le test, sauvegarder avec `joblib`.

## Résultats

| Modèle | RMSE en validation croisée |
|---|---|
| Régression linéaire | ~70 000 $ (sous-apprentissage) |
| Arbre de décision | ~67 000 $ (sur-apprentissage : 0 $ sur le train) |
| Forêt aléatoire | ~47 100 $ |
| Gradient boosting (HGB, en bonus) | ~45 400 $ |
| Forêt aléatoire réglée (`RandomizedSearchCV`) | ~42 100 $ |

Sur le jeu de test, modèle final : **RMSE ≈ 41 400 $**, intervalle de confiance à 95 % entre 39 500 $ et 43 700 $.

La variable la plus importante c'est le revenu médian (~19 %), ensuite le fait d'être dans les terres (`INLAND`).

## À retenir

- Mettre le test de côté avant d'explorer, sinon je triche sans le savoir (*data snooping*).
- Stratifier le découpage quand le jeu de données est petit.
- `fit` / `transform` / `predict`, et les pipelines pour éviter les fuites de données entre train et test.
- La validation croisée montre si je suis en sous- ou sur-apprentissage.
- On peut régler le prétraitement et le modèle en même temps (`preprocessing__geo__n_clusters`).

## La suite

- Les exercices du chapitre : SVR, `SelectFromModel`, transformateur KNN perso.
- Refaire le réglage avec `HistGradientBoostingRegressor`.
- Mettre le modèle derrière une petite API FastAPI.
