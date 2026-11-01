# 🏠 Projet 1 — California Housing Prices Prediction

> Chapitre 2 de *Hands-On Machine Learning* (Aurélien Géron) — « End-to-End Machine Learning Project »

## 🎯 Objectif
Prédire le **prix médian des logements** d'un district californien (recensement de 1990) à partir de
caractéristiques comme le revenu médian, la population ou la localisation.
Type de problème : **régression supervisée**, métrique : **RMSE**.

## 📂 Contenu du dossier
| Fichier | Rôle |
|---|---|
| `california_housing_complet.ipynb` | **Le projet complet, expliqué pas à pas pour débutant** : pour chaque étape, *pourquoi*, *méthode choisie*, *alternatives* et *pourquoi pas l'alternative* |
| `housing.ipynb` | Mon premier essai (exploration initiale) |
| `datasets/` | Créé automatiquement par le notebook (téléchargement des données) — ignoré par Git |
| `models/` | Modèle entraîné (`.pkl`, ~150 Mo) — ignoré par Git |

## ▶️ Lancer le projet
```bash
cd machine-learning-specialization/Hands_On_Machine_Learning_Projects
pip install -r requirements.txt
jupyter notebook California_Housing_Prices_Prediction/california_housing_complet.ipynb
```
Exécution complète : ~5 minutes (la recherche d'hyperparamètres est l'étape la plus longue).

## 🧭 Démarche
1. Cadrer le problème et choisir la métrique (RMSE)
2. Télécharger et inspecter les données (valeurs manquantes, plafonds, distributions asymétriques)
3. Créer un jeu de test **stratifié** sur le revenu (20 %)
4. Explorer : carte géographique, corrélations, nouveaux ratios (`bedrooms_ratio`, `rooms_per_house`, `people_per_house`)
5. Préparer les données dans un **pipeline** Scikit-Learn : imputation, one-hot, log, standardisation, similarité à 10 clusters géographiques (transformateur personnalisé K-means + RBF) → 24 features
6. Comparer des modèles par **validation croisée** (10 plis)
7. Affiner la forêt aléatoire (`GridSearchCV`, puis `RandomizedSearchCV`), analyser importances et erreurs par groupe
8. Évaluer une seule fois sur le jeu de test, puis sauvegarder le modèle (`joblib`)

## 📊 Résultats
| Modèle | RMSE (validation croisée) |
|---|---|
| Régression linéaire | ~70 000 $ (sous-apprentissage) |
| Arbre de décision | ~67 000 $ (sur-apprentissage : 0 $ sur le train) |
| Forêt aléatoire | ~47 100 $ |
| Gradient Boosting (HGB, bonus) | ~45 400 $ |
| Forêt aléatoire optimisée (`RandomizedSearchCV`) | ~42 100 $ |

**Jeu de test (modèle final) : RMSE ≈ 41 400 $** — intervalle de confiance à 95 % : [39 500 $ ; 43 700 $].

Feature la plus importante : **le revenu médian** (~19 % de l'importance), puis le fait d'être à l'intérieur des terres (`INLAND`).

## 💡 Ce que j'ai appris
- Mettre le jeu de test de côté **avant** d'explorer (éviter le *data snooping*)
- Pourquoi stratifier un découpage sur un petit jeu de données
- Le cycle `fit` / `transform` / `predict` et les **pipelines** pour éviter les fuites de données
- Diagnostiquer sous-apprentissage vs sur-apprentissage avec la validation croisée
- Régler prétraitement et modèle en même temps (`preprocessing__geo__n_clusters`)

## 🏋️ Pistes / exercices suivants
- Exercices du chapitre : SVR, `SelectFromModel`, transformateur KNN personnalisé
- Refaire l'optimisation avec `HistGradientBoostingRegressor`
- Exposer le modèle via une petite API (FastAPI)
