# IA / Machine learning

Mes notes de cours et mes exercices en IA et machine learning. Je suis le parcours d'Andrew Ng (DeepLearning.AI sur Coursera), et le livre *Hands-On Machine Learning* d'Aurélien Géron pour la pratique. Le principe : coder les algos à la main avant d'utiliser une bibliothèque.

## Ce qu'il y a

| Dossier | Contenu | État |
|---|---|---|
| `machine-learning-specialization/ft_linear_regression/` | Régression linéaire codée à la main, descente de gradient | fait |
| `machine-learning-specialization/Hands_On_Machine_Learning_Projects/California_Housing_Prices_Prediction/` | Projet complet du chapitre 2 de Hands-On ML | fait |
| `ai-for-everyone-notes/` | Notes du cours AI For Everyone | vide |
| `deep-learning-specialization/` | Deep Learning Specialization | vide |
| `generative-ai-notes/` | Generative AI for Everyone | vide |
| `projects/rag-demo/` | Un petit RAG | à faire |
| `projects/agent-demo/` | Un agent simple avec appel d'outils | à faire |

## ft_linear_regression

Prédire le prix d'une voiture à partir de son kilométrage (`dataset/data.csv`). Lecture du CSV, nuage de points, hypothèse `prix = θ0 + θ1 × km`, fonction de coût, puis descente de gradient codée à la main.

Le problème que j'ai eu : sans normaliser les données, le kilométrage est tellement grand que le taux d'apprentissage doit descendre vers **1e-11** pour que ça ne diverge pas. La suite logique : normaliser, et comparer avec scikit-learn.

## California Housing

Voir le README du dossier. RMSE finale sur le jeu de test : environ 41 400 $.

## Lancer

Les notebooks ont leurs dépendances dans `machine-learning-specialization/Hands_On_Machine_Learning_Projects/requirements.txt` (jupyter, numpy, pandas, matplotlib, scikit-learn, scipy, joblib).

```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -r machine-learning-specialization/Hands_On_Machine_Learning_Projects/requirements.txt
jupyter notebook
```

## Où j'en suis

- [x] Régression linéaire avec descente de gradient
- [x] California Housing de bout en bout
- [ ] Normaliser ft_linear_regression et comparer avec scikit-learn
- [ ] AI For Everyone
- [ ] Machine Learning Specialization
- [ ] Deep Learning Specialization
- [ ] Generative AI for Everyone
- [ ] Premier RAG qui marche
- [ ] Premier agent qui marche

Licence MIT.
