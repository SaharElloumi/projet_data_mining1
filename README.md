# Explainable Predictive Maintenance System (AI4I 2020)

Projet Challenge 1 — Data Mining · II3-IA / II3-IF · ENSI · 2026-2027
Réalisé par : **Prénom NOM** — Cas n°15 : Industrie – Maintenance prédictive (UCI AI4I 2020, cible : `Machine failure`)

- Application déployée : <LIEN STREAMLIT COMMUNITY CLOUD>
- API FastAPI (docs) : <LIEN API>/docs
- One-pager : `ONE_PAGER.md` → exporter en PDF

## Structure
```
app.py          # Application Streamlit (9 pages : Accueil → Explainable AI)
core.py         # Données, préparation, modèles, métriques, aides SHAP/LIME (partagé)
api/main.py     # API FastAPI : /health, /model-info, /predict, /explain
notebook.ipynb  # Notebook d'exploration (à ajouter)
requirements.txt
```
## Lancement local
```bash
python -m pip install -r requirements.txt 
python -m uvicorn api.main:app --port 8000         # terminal 1
python -m streamlit run app.py                      # terminal 2  (API_URL=http://localhost:8000 par défaut)
```
Le jeu de données est téléchargé automatiquement via `ucimlrepo` (id 601). Si l'API est injoignable, l'application bascule sur le modèle de référence local.

## Déploiement
1. API : Hugging Face Spaces (Docker) ou Render, commande `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
2. App : Streamlit Community Cloud sur ce dépôt, `app.py`, puis variable secrète `API_URL=<url de l'API>`.

## Captures d'écran
Ajouter dans `docs/` les captures de chaque page et les référencer ici.
