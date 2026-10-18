# One-pager — Maintenance prédictive explicable (AI4I 2020)
**Prénom NOM · II3-IA/IF · ENSI 2026-2027**

**Problème métier.** Anticiper une panne machine pour planifier la maintenance et éviter les arrêts de production. Une fausse alerte coûte une inspection ; une panne manquée coûte un arrêt.

**Données.** AI4I 2020 (UCI) : 10 000 observations, 6 variables (Type, températures air/process, vitesse de rotation, couple, usure d'outil), 339 pannes (3,39 %). Aucune valeur manquante ni doublon. Split stratifié 80/20 (random_state=42).

**Meilleur modèle.** XGBoost (PR-AUC 0,810 ; ROC-AUC 0,970 ; rappel 0,824 au seuil 0,50). Random Forest : meilleur F1 (0,721). Régression logistique : meilleur rappel mais précision 0,142. Seuil retenu pour la référence API : 0,75. *(Compléter avec vos chiffres finaux.)*

**Explications clés.** Permutation : le couple (Torque) arrive en tête ; SHAP : l'usure d'outil et la température de l'air pèsent le plus. Une usure élevée, un couple élevé et une faible vitesse de rotation augmentent le risque. *(Vérifier avec vos graphiques.)*

**Analyse critique.**
- Cohérence SHAP / LIME : <à compléter — accord de signe, top-3 commun, cohérence avec la physique : couple × vitesse = puissance, usure d'outil>.
- Variables sensibles : aucune (données machine) ; seul `Type` (L/M/H) est une catégorie de produit, impact d'équité nul.
- Performance vs interprétabilité : XGBoost + SHAP pour la détection ; régression logistique comme repère auditable. Recommandation : XGBoost au seuil ajusté au coût métier.
- Actions : remplacer l'outil avant un seuil d'usure critique, limiter le couple en haute charge, surveiller la température de l'air.

**Limites.** Jeu synthétique, très déséquilibré, peu de pannes en test (68) → métriques instables ; SHAP et LIME décrivent le modèle, pas une causalité.
