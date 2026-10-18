"""Noyau commun : données, préparation, modèles, métriques, aides XAI.
Utilisé par l'application Streamlit (app.py) ET par l'API FastAPI (api/main.py)."""
import os
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(ROOT, "models", "ai4i2020.csv")
REF_PATH = os.path.join(ROOT, "models", "xgb_reference.joblib")

TARGET = "Machine failure"
FEATURES = ["Type", "Air temperature", "Process temperature",
            "Rotational speed", "Torque", "Tool wear"]
THRESHOLD_REF = 0.75
SEED = 42

# --------------------------------------------------------------------- Données
def load_data() -> pd.DataFrame:
    """Téléchargement automatique AI4I 2020 (UCI id=601), mis en cache en CSV."""
    if os.path.exists(DATA_CSV):
        return pd.read_csv(DATA_CSV)
    from ucimlrepo import fetch_ucirepo
    d = fetch_ucirepo(id=601)
    df = pd.concat([d.data.features, d.data.targets], axis=1)
    df.columns = [c.split(" [")[0] for c in df.columns]
    df = df[FEATURES + [TARGET]].copy()
    os.makedirs(os.path.dirname(DATA_CSV), exist_ok=True)
    df.to_csv(DATA_CSV, index=False)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Imputation : médiane (numérique) / mode (catégoriel)."""
    df = df.copy()
    for c in df.columns:
        if df[c].isna().any():
            num = pd.api.types.is_numeric_dtype(df[c])
            df[c] = df[c].fillna(df[c].median() if num else df[c].mode()[0])
    return df


def encode(X: pd.DataFrame, cols=None) -> pd.DataFrame:
    """One-hot des variables catégorielles ; `cols` impose l'ordre/colonnes de l'entraînement."""
    cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
    out = pd.get_dummies(X, columns=cat, dtype=float).astype(float)
    return out.reindex(columns=cols, fill_value=0.0) if cols is not None else out


def orig_of(col: str, feats) -> str:
    """Variable d'origine d'une colonne encodée (ex. Type_L -> Type)."""
    return max([f for f in feats if col == f or col.startswith(f + "_")], key=len)


def split(df, target, features, test_size=0.2):
    d = clean(df[list(features) + [target]])
    y = d[target].astype(int)
    X = encode(d[list(features)])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size,
                                          stratify=y, random_state=SEED)
    return dict(features=list(features), target=target, test_size=test_size,
                cols=list(X.columns), Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte,
                raw_te=d[list(features)].loc[Xte.index])

# --------------------------------------------------------------------- Modèles
MODELS = ["Logistic Regression", "Decision Tree", "KNN", "Random Forest", "XGBoost"]
SCALED = ("Logistic Regression", "KNN")

DEFAULTS = {
    "Logistic Regression": dict(C=1, solver="lbfgs", class_weight="balanced"),
    "Decision Tree": dict(max_depth=None, min_samples_split=2, min_samples_leaf=2,
                          class_weight="balanced"),
    "KNN": dict(n_neighbors=3, weights="distance", p=2),
    "Random Forest": dict(n_estimators=100, max_depth=None, min_samples_split=5,
                          min_samples_leaf=2, max_features="log2", class_weight="balanced"),
    "XGBoost": dict(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                    scale_pos_weight="calculé sur le train"),
}
GRIDS = {  # pour l'optimisation RandomizedSearchCV
    "Logistic Regression": dict(C=[0.01, 0.1, 1, 10, 100]),
    "Decision Tree": dict(max_depth=[3, 5, 8, 12, None], min_samples_leaf=[1, 2, 5, 10]),
    "KNN": dict(n_neighbors=[3, 5, 7, 11, 15], weights=["uniform", "distance"]),
    "Random Forest": dict(n_estimators=[100, 200, 300], max_depth=[None, 8, 12],
                          min_samples_leaf=[1, 2, 4]),
    "XGBoost": dict(n_estimators=[100, 200, 300], max_depth=[3, 4, 6],
                    learning_rate=[0.03, 0.05, 0.1]),
}


def build_model(name, params, ytr):
    p = dict(params)
    if name == "Logistic Regression":
        return Pipeline([("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, random_state=SEED, **p))])
    if name == "KNN":
        return Pipeline([("sc", StandardScaler()), ("clf", KNeighborsClassifier(**p))])
    if name == "Decision Tree":
        return DecisionTreeClassifier(random_state=SEED, **p)
    if name == "Random Forest":
        return RandomForestClassifier(n_jobs=-1, random_state=SEED, **p)
    spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    p["scale_pos_weight"] = spw
    return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, **p)


def fit_model(name, params, Xtr, ytr, tune=False):
    m = build_model(name, params, ytr)
    if tune:
        pre = "clf__" if name in SCALED else ""
        grid = {pre + k: v for k, v in GRIDS[name].items()}
        s = RandomizedSearchCV(m, grid, n_iter=6, scoring="average_precision",
                               cv=3, random_state=SEED, n_jobs=-1)
        s.fit(Xtr, ytr)
        best = {k.replace(pre, ""): v for k, v in s.best_params_.items()}
        return s.best_estimator_, {**params, **best}
    return m.fit(Xtr, ytr), params


def proba(model, X):
    return model.predict_proba(X)[:, 1]


def metrics(y, p, thr=0.5):
    yh = (np.asarray(p) >= thr).astype(int)
    return dict(precision=precision_score(y, yh, zero_division=0),
                recall=recall_score(y, yh, zero_division=0),
                f1=f1_score(y, yh, zero_division=0),
                roc_auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p))

# ----------------------------------------------------------------- Référence API
def load_reference():
    """Modèle XGBoost pré-entraîné (seuil 0,75) partagé entre l'API et l'application."""
    if os.path.exists(REF_PATH):
        return joblib.load(REF_PATH)
    cfg = split(load_data(), TARGET, FEATURES, 0.2)
    model, _ = fit_model("XGBoost", DEFAULTS["XGBoost"], cfg["Xtr"], cfg["ytr"])
    bundle = dict(model=model, cfg=cfg, cols=cfg["cols"], threshold=THRESHOLD_REF)
    os.makedirs(os.path.dirname(REF_PATH), exist_ok=True)
    joblib.dump(bundle, REF_PATH)
    return bundle

# --------------------------------------------------------------------- XAI
def native_importance(name, model, cols):
    m = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if hasattr(m, "feature_importances_"):
        return pd.Series(m.feature_importances_, index=cols)
    if hasattr(m, "coef_"):
        return pd.Series(np.abs(m.coef_[0]), index=cols)
    return None  # KNN : pas d'importance native


def group_series(s: pd.Series, feats) -> pd.Series:
    g = {}
    for c, v in s.items():
        f = orig_of(c, feats)
        g[f] = g.get(f, 0) + v
    return pd.Series(g)


def group_values(sv, cols, feats) -> pd.DataFrame:
    g = {f: np.zeros(sv.shape[0]) for f in feats}
    for j, c in enumerate(cols):
        g[orig_of(c, feats)] += sv[:, j]
    return pd.DataFrame(g)


def grouped_permutation(model, X, y, cols, feats, n_repeats=5):
    """Importance par permutation (baisse d'Average Precision), colonnes one-hot permutées ensemble."""
    rng = np.random.RandomState(SEED)
    base = average_precision_score(y, proba(model, X))
    rows = {}
    for f in feats:
        cs = [c for c in cols if orig_of(c, feats) == f]
        d = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[cs] = X[cs].values[rng.permutation(len(X))]
            d.append(base - average_precision_score(y, proba(model, Xp)))
        rows[f] = (np.mean(d), np.std(d))
    return pd.DataFrame(rows, index=["importance_mean", "importance_std"]).T


def shap_values(name, model, X, bg):
    """Retourne (valeurs SHAP [n,p], valeur de base)."""
    if name in ("Decision Tree", "Random Forest", "XGBoost"):
        ex = shap.TreeExplainer(model)
        sv = ex.shap_values(X)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1]
        ev = np.atleast_1d(ex.expected_value)
        return sv, float(ev[-1])
    f = lambda d: model.predict_proba(pd.DataFrame(d, columns=X.columns))[:, 1]
    e = shap.Explainer(f, shap.sample(bg, 100, random_state=SEED))(X)
    return e.values, float(np.mean(e.base_values))


def sentence(contrib: pd.Series, p: float, thr: float, label="panne") -> str:
    """Phrase de synthèse automatique pour un décideur non spécialiste."""
    if p >= thr:
        top = [f for f, v in contrib.sort_values(ascending=False).items() if v > 0][:2]
        if not top:
            return f"Le risque de {label} est élevé ({p:.1%}) sans facteur dominant."
        return f"Le risque de {label} est élevé ({p:.1%}) principalement à cause de " + " et ".join(top) + "."
    top = list(contrib.sort_values().index[:2])
    return f"Le risque de {label} est faible ({p:.1%}) principalement grâce à " + " et ".join(top) + "."
