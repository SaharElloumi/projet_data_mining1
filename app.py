"""Explainable Predictive Maintenance System — AI4I 2020 (UCI)
Projet Data Mining · II3-IA / II3-IF · ENSI · 2026-2027"""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

import core

AUTHOR = "Sahar Elloumi"            # <-- À MODIFIER avec vos coordonnées
try:
    API_URL = os.getenv("API_URL") or st.secrets["API_URL"]
except Exception:
    API_URL = "http://localhost:8000"
POS, NEG = "panne", "pas de panne"
REF = "Modèle de référence pré-entraîné (API)"
PAGES = ["Accueil", "Données", "Exploration", "Préparation", "Modélisation",
         "Comparaison", "Évaluation", "Prédiction", "Explainable AI"]

st.set_page_config(page_title="Projet Challenge 1", page_icon="👩🏻‍💼", layout="wide")
st.markdown("""<style>
.banner{background:linear-gradient(135deg,#1b3a57,#2d5a80);padding:2.2rem 2.5rem;
border-radius:14px;margin-bottom:1.2rem}.banner h1{margin:0;color:white}
.banner p{margin:.4rem 0 0;color:#cfe0ef}</style>""", unsafe_allow_html=True)

ss = st.session_state


@st.cache_data(show_spinner="Téléchargement du jeu de données AI4I 2020…")
def default_df():
    return core.load_data()


@st.cache_resource(show_spinner="Chargement du modèle de référence…")
def reference():
    return core.load_reference()


ss.setdefault("df", default_df())
ss.setdefault("target", core.TARGET)
ss.setdefault("dsname", "AI4I 2020")
ss.setdefault("models", {})
ss.setdefault("xai", {})
ss.setdefault("local", None)
if "cfg" not in ss:
    ss.cfg = core.split(ss.df, ss.target, core.FEATURES, 0.2)
df, target = ss.df, ss.target


def metric(col, label, value):
    col.metric(label, value, border=True)


def cm_fig(y, p, thr, title=""):
    tn, fp, fn, tp = confusion_matrix(y, (np.asarray(p) >= thr).astype(int), labels=[0, 1]).ravel()
    fig = go.Figure(go.Heatmap(
        z=[[tn, fp], [fn, tp]], x=[f"Prédit : {NEG}", f"Prédit : {POS}"],
        y=[f"Réel : {NEG}", f"Réel : {POS}"],
        text=[[f"TN<br>{tn}", f"FP<br>{fp}"], [f"FN<br>{fn}", f"TP<br>{tp}"]],
        texttemplate="%{text}", colorscale="Blues", showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_dark", title=title, height=300, margin=dict(t=40, b=10))
    return fig


# ------------------------------------------------------------------ Barre latérale
st.sidebar.title("Projet Challenge 1")
page = st.sidebar.radio("Navigation", PAGES, key="page", label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.metric("Modèles en session", len(ss.models), border=True)
st.sidebar.caption(f"Référence API : XGBoost pré-entraîné · seuil {core.THRESHOLD_REF:.2f}".replace(".", ","))

# ===================================================================== ACCUEIL
if page == "Accueil":
    st.markdown(f"""<div class="banner"><h1>Explainable Predictive Maintenance System</h1>
    <p>Projet Data Mining · II3-IA / II3-IF · ENSI · 2026-2027 · Réalisé par {AUTHOR}</p></div>""",
                unsafe_allow_html=True)
    n, k = len(df), int(df[target].sum())
    c = st.columns(4)
    metric(c[0], "Dataset", ss.dsname)
    metric(c[1], "Observations", f"{n:,}".replace(",", " "))
    metric(c[2], "Variables explicatives", df.shape[1] - 1)
    metric(c[3], "Pannes", f"{k} ({k / n:.2%})")
    st.write(f"{n:,} observations {df.shape[1] - 1} variables explicatives et {k} pannes ({k / n:.2%}).".replace(",", " "))

# ===================================================================== DONNÉES
elif page == "Données":
    st.title("Données")
    c = st.columns(4)
    metric(c[0], "Lignes", f"{len(df):,}".replace(",", " "))
    metric(c[1], "Colonnes", df.shape[1])
    metric(c[2], "Valeurs manquantes", int(df.isna().sum().sum()))
    metric(c[3], "Doublons", int(df.duplicated().sum()))
    t1, t2, t3 = st.tabs(["Aperçu", "Statistiques", "Qualité"])
    with t1:
        st.dataframe(df.head(100), use_container_width=True)
    with t2:
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    with t3:
        q = pd.DataFrame({"Type": df.dtypes.astype(str), "Manquantes": df.isna().sum(),
                          "Valeurs uniques": df.nunique()})
        st.dataframe(q, use_container_width=True)
        vc = df[target].value_counts().rename({0: NEG, 1: POS}).reset_index()
        vc.columns = ["Classe", "Effectif"]
        st.plotly_chart(px.bar(vc, x="Classe", y="Effectif", text="Effectif", template="plotly_dark",
                               title="Proportion de chaque classe"), use_container_width=True)
    with st.expander("Importer mon propre fichier CSV"):
        up = st.file_uploader("Fichier CSV", type="csv")
        if up is not None:
            new = pd.read_csv(up)
            tg = st.selectbox("Variable cible (binaire)", new.columns)
            if st.button("Utiliser ce fichier"):
                vals = sorted(new[tg].dropna().unique())
                if len(vals) != 2:
                    st.error("La cible doit être binaire (2 classes).")
                else:
                    new[tg] = (new[tg] == vals[1]).astype(int)
                    ss.df, ss.target, ss.dsname = new, tg, up.name
                    ss.models, ss.xai = {}, {}
                    ss.cfg = core.split(new, tg, [c for c in new.columns if c != tg], 0.2)
                    st.rerun()
        if ss.dsname != "AI4I 2020" and st.button("Revenir à AI4I 2020"):
            for k in ("df", "target", "dsname", "cfg"):
                ss.pop(k, None)
            ss.models, ss.xai = {}, {}
            st.rerun()

# ================================================================== EXPLORATION
elif page == "Exploration":
    st.title("Exploration")
    num = [c for c in df.columns if c != target and pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in df.columns if c != target and not pd.api.types.is_numeric_dtype(df[c])]
    v = st.selectbox("Variable numérique", num)
    d = df.assign(Statut=df[target].map({0: "Pas de panne", 1: "Panne"}))
    st.plotly_chart(px.box(d, x="Statut", y=v, color="Statut", template="plotly_dark",
                           color_discrete_map={"Pas de panne": "#4c8bbd", "Panne": "#d62728"},
                           title=f"{v} selon l'état de la machine"), use_container_width=True)
    m0, m1 = df.loc[df[target] == 0, v].median(), df.loc[df[target] == 1, v].median()
    st.write(f"Médiane sans panne : {m0:.2f} · avec panne : {m1:.2f}. Les intervalles interquartiles se chevauchent.")
    c1, c2 = st.columns(2)
    if cat:
        cv = c1.selectbox("Variable catégorielle", cat) if len(cat) > 1 else cat[0]
        r = df.groupby(cv)[target].mean().reset_index(name="Taux de panne")
        c1.plotly_chart(px.bar(r, x=cv, y="Taux de panne", color="Taux de panne", text_auto=".2%",
                               color_continuous_scale="Blues", template="plotly_dark",
                               title=f"Taux de panne observé selon {cv}"), use_container_width=True)
    corr = df[num + [target]].corr()
    c2.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1, color_continuous_scale="RdBu_r",
                              template="plotly_dark", title="Matrice de corrélation numérique"),
                    use_container_width=True)

# ================================================================== PRÉPARATION
elif page == "Préparation":
    st.title("Préparation")
    tg = st.selectbox("Variable cible", df.columns, index=list(df.columns).index(target))
    opts = [c for c in df.columns if c != tg]
    default = [f for f in ss.cfg["features"] if f in opts] or opts
    feats = st.multiselect("Variables explicatives", opts, default=default)
    ts = st.slider("Part du jeu de test", 0.10, 0.40, float(ss.cfg["test_size"]), 0.05)
    st.caption("Traitement automatique : imputation (médiane/mode), encodage one-hot, "
               "normalisation (StandardScaler) intégrée aux pipelines de régression logistique et KNN.")
    if st.button("Enregistrer la configuration", type="primary"):
        if not feats:
            st.error("Sélectionnez au moins une variable.")
        else:
            ss.target = tg
            ss.cfg = core.split(df, tg, feats, ts)
            st.success("Configuration enregistrée. Les modèles existants ne sont pas modifiés.")
    cfg = ss.cfg
    c = st.columns(3)
    metric(c[0], "Apprentissage", len(cfg["Xtr"]))
    metric(c[1], "Test", len(cfg["Xte"]))
    metric(c[2], "Variables", len(cfg["features"]))
    st.write(f"{len(cfg['Xtr']):,} observations pour l'apprentissage et {len(cfg['Xte']):,} pour le test · "
             f"test_size={cfg['test_size']:.2f} · split stratifié · random_state=42".replace(",", " "))

# ================================================================= MODÉLISATION
elif page == "Modélisation":
    cfg = ss.cfg
    st.title("Modélisation")
    st.caption(f"Configuration active : {len(cfg['features'])} variable(s), test_size={cfg['test_size']:.2f}.")
    chosen = st.multiselect("Modèles à entraîner", core.MODELS, default=core.MODELS)
    mode = st.radio("Mode de configuration", ["Paramètres optimisés recommandés", "Configuration manuelle"],
                    horizontal=True)
    tune = False
    params = {}
    if mode.startswith("Paramètres"):
        tune = st.checkbox("Affiner par RandomizedSearchCV (plus lent)")
        for m in chosen:
            with st.expander(f"⚙ {m}", expanded=True):
                st.json(core.DEFAULTS[m])
            params[m] = dict(core.DEFAULTS[m])
    else:
        for m in chosen:
            with st.expander(f"⚙ {m}", expanded=True):
                k = lambda s: f"{m}_{s}"
                c = st.columns(3)
                if m == "Logistic Regression":
                    params[m] = dict(C=c[0].number_input("C", 0.001, 100.0, 1.0, key=k("C")), solver="lbfgs",
                                     class_weight=c[1].selectbox("class_weight", ["balanced", None], key=k("cw")))
                elif m == "Decision Tree":
                    md = c[0].slider("max_depth (0 = illimité)", 0, 30, 0, key=k("md"))
                    params[m] = dict(max_depth=md or None,
                                     min_samples_split=c[1].slider("min_samples_split", 2, 20, 2, key=k("ss")),
                                     min_samples_leaf=c[2].slider("min_samples_leaf", 1, 20, 2, key=k("sl")),
                                     class_weight="balanced")
                elif m == "KNN":
                    params[m] = dict(n_neighbors=c[0].slider("n_neighbors", 1, 30, 3, key=k("n")),
                                     weights=c[1].selectbox("weights", ["distance", "uniform"], key=k("w")),
                                     p=c[2].selectbox("p", [1, 2], index=1, key=k("p")))
                elif m == "Random Forest":
                    md = c[0].slider("max_depth (0 = illimité)", 0, 30, 0, key=k("md"))
                    params[m] = dict(n_estimators=c[1].slider("n_estimators", 50, 500, 100, 50, key=k("ne")),
                                     max_depth=md or None,
                                     min_samples_split=c[2].slider("min_samples_split", 2, 20, 5, key=k("ss")),
                                     min_samples_leaf=c[0].slider("min_samples_leaf", 1, 20, 2, key=k("sl")),
                                     max_features=c[1].selectbox("max_features", ["log2", "sqrt"], key=k("mf")),
                                     class_weight="balanced")
                else:
                    params[m] = dict(n_estimators=c[0].slider("n_estimators", 50, 500, 200, 50, key=k("ne")),
                                     max_depth=c[1].slider("max_depth", 2, 10, 4, key=k("md")),
                                     learning_rate=c[2].number_input("learning_rate", 0.01, 0.5, 0.1, 0.01, key=k("lr")),
                                     subsample=c[0].slider("subsample", 0.5, 1.0, 0.8, 0.1, key=k("sub")))
    st.subheader("Configuration sélectionnée")
    for m in chosen:
        st.markdown(f"**{m}**")
        st.json(params[m], expanded=False)
    if st.button("Entraîner les modèles sélectionnés", type="primary", disabled=not chosen):
        bar = st.progress(0.0, text="Entraînement en cours…")
        for i, m in enumerate(chosen):
            model, used = core.fit_model(m, params[m], cfg["Xtr"], cfg["ytr"], tune)
            ss.models[m] = dict(model=model, params=used, cfg=cfg, y=cfg["yte"],
                                proba=core.proba(model, cfg["Xte"]))
            ss.xai = {k: v for k, v in ss.xai.items() if k != m}
            bar.progress((i + 1) / len(chosen), text=f"{m} entraîné")
        bar.progress(1.0, text="Entraînement terminé")
        st.success(f"{len(chosen)} modèle(s) entraîné(s) et conservé(s) dans la session.")
        st.info("Modèles disponibles dans la session : " + ", ".join(ss.models))

# ================================================================== COMPARAISON
elif page == "Comparaison":
    st.title("Comparaison")
    if not ss.models:
        st.info("Entraînez d'abord des modèles dans la page Modélisation.")
        st.stop()
    rows = {n: core.metrics(m["y"], m["proba"], 0.5) for n, m in ss.models.items()}
    T = pd.DataFrame(rows).T
    T.columns = ["Précision panne", "Rappel panne", "F1 panne", "ROC-AUC", "PR-AUC (Average Precision)"]
    T.index.name = "Modèle"
    st.dataframe(T.round(3), use_container_width=True)
    st.write(f"Meilleur F1-score : {T['F1 panne'].idxmax()} ({T['F1 panne'].max():.3f}) · "
             f"meilleur rappel : {T['Rappel panne'].idxmax()} ({T['Rappel panne'].max():.3f}) · "
             f"meilleure PR-AUC : {T['PR-AUC (Average Precision)'].idxmax()} ({T['PR-AUC (Average Precision)'].max():.3f}).")
    with st.expander("Voir les hyperparamètres"):
        for n, m in ss.models.items():
            st.markdown(f"**{n}**")
            st.json(m["params"], expanded=False)
    mc = st.selectbox("Métrique du graphique", list(T.columns))
    st.plotly_chart(px.bar(T.reset_index(), x="Modèle", y=mc, color="Modèle", text_auto=".3f",
                           template="plotly_dark", title=f"Comparaison — {mc}"), use_container_width=True)
    c1, c2 = st.columns(2)
    roc, pr = go.Figure(), go.Figure()
    for n, m in ss.models.items():
        fpr, tpr, _ = roc_curve(m["y"], m["proba"])
        pcs, rcl, _ = precision_recall_curve(m["y"], m["proba"])
        roc.add_scatter(x=fpr, y=tpr, name=f"{n} (AUC={rows[n]['roc_auc']:.3f})")
        pr.add_scatter(x=rcl, y=pcs, name=f"{n} (AP={rows[n]['pr_auc']:.3f})")
    roc.add_scatter(x=[0, 1], y=[0, 1], name="Aléatoire", line=dict(dash="dash", color="grey"))
    roc.update_layout(template="plotly_dark", title="Courbes ROC", xaxis_title="Taux de faux positifs",
                      yaxis_title="Taux de vrais positifs")
    pr.update_layout(template="plotly_dark", title="Courbes précision-rappel — AP dans la légende",
                     xaxis_title="Rappel", yaxis_title="Précision")
    c1.plotly_chart(roc, use_container_width=True)
    c2.plotly_chart(pr, use_container_width=True)
    st.subheader("Matrices de confusion au seuil 0,50")
    cols = st.columns(2)
    for i, (n, m) in enumerate(ss.models.items()):
        cols[i % 2].plotly_chart(cm_fig(m["y"], m["proba"], 0.5, n), use_container_width=True)

# =================================================================== ÉVALUATION
elif page == "Évaluation":
    st.title("Évaluation")
    if not ss.models:
        st.info("Entraînez d'abord des modèles dans la page Modélisation.")
        st.stop()
    n = st.selectbox("Modèle", list(ss.models))
    thr = st.slider("Seuil de décision", 0.0, 1.0, 0.5, 0.01)
    m = ss.models[n]
    r = core.metrics(m["y"], m["proba"], thr)
    c = st.columns(4)
    metric(c[0], "Précision panne", f"{r['precision']:.3f}")
    metric(c[1], "Rappel panne", f"{r['recall']:.3f}")
    metric(c[2], "F1-score", f"{r['f1']:.3f}")
    metric(c[3], "Seuil", f"{thr:.2f}")
    st.write(f"Au seuil {thr:.2f} : précision {r['precision']:.1%}, rappel {r['recall']:.1%}, F1-score {r['f1']:.1%}. "
             f"ROC-AUC = {r['roc_auc']:.3f} · PR-AUC = {r['pr_auc']:.3f}.")
    st.plotly_chart(cm_fig(m["y"], m["proba"], thr, f"{n} — seuil {thr:.2f}"), use_container_width=True)

# =================================================================== PRÉDICTION
elif page == "Prédiction":
    st.title("Prédiction")
    choice = st.selectbox("Modèle utilisé", [REF] + list(ss.models))
    is_ref = choice == REF
    src = default_df() if is_ref else ss.models[choice]["cfg"]["raw_te"]
    feats = core.FEATURES if is_ref else ss.models[choice]["cfg"]["features"]
    ref_defaults = dict(zip(core.FEATURES, ["L", 300.0, 310.0, 1500, 40.0, 100]))
    rec = {}
    with st.container(border=True):
        cols = st.columns(2)
        for i, f in enumerate(feats):
            col = cols[i % 2]
            if is_ref and f == "Type" or not pd.api.types.is_numeric_dtype(src[f]):
                opts = sorted(src[f].astype(str).unique())
                rec[f] = col.selectbox(f, opts, index=opts.index(ref_defaults[f]) if is_ref and ref_defaults[f] in opts else 0, key=f"p_{f}")
            else:
                dv = ref_defaults[f] if is_ref else float(src[f].median())
                rec[f] = col.number_input(f, value=type(dv)(dv), key=f"p_{f}")
        go_btn = st.button("Analyser la machine", type="primary")
    if go_btn:
        ss.pred = dict(choice=choice, rec=rec)
    if ss.get("pred") and ss.pred["choice"] == choice:
        rec = ss.pred["rec"]
        if is_ref:
            try:
                r = requests.post(f"{API_URL}/predict", json=rec, timeout=10)
                r.raise_for_status()
                out = r.json()
                score, thr = out["score"], out["threshold"]
                src_txt = "API FastAPI"
            except Exception:
                b = reference()
                score = float(core.proba(b["model"], core.encode(pd.DataFrame([rec])[core.FEATURES], b["cols"]))[0])
                thr, src_txt = b["threshold"], "modèle local (API indisponible)"
        else:
            m = ss.models[choice]
            x = core.encode(pd.DataFrame([rec])[feats], m["cfg"]["cols"])
            score, thr, src_txt = float(core.proba(m["model"], x)[0]), 0.5, "modèle de la session"
        fail = score >= thr
        c = st.columns(3)
        metric(c[0], "Décision", "Panne" if fail else "Pas de panne")
        metric(c[1], "Score de risque du modèle", f"{score:.3f}")
        metric(c[2], "Seuil utilisé", f"{thr:.2f}")
        st.caption(f"Modèle utilisé : {choice} · source : {src_txt}")
        st.write(f"Score = {score:.3f} {'≥' if fail else '<'} seuil = {thr:.2f} → {'Panne' if fail else 'Pas de panne'}")
        (st.error if fail else st.success)(
            "Une panne est signalée pour les mesures saisies." if fail else "Aucune panne n'est signalée pour les mesures saisies.")

        def goto():
            ss.local = dict(choice=choice, rec=rec, feats=feats)
            ss.page = "Explainable AI"
        st.button("Expliquer cette prédiction", on_click=goto)

# ================================================================ EXPLAINABLE AI
else:
    st.title("Explainable AI")
    choices = [REF.replace("Modèle de référence pré-entraîné (API)", "XGBoost — modèle de référence")] + list(ss.models)
    pre = 0
    if ss.local:
        want = "XGBoost — modèle de référence" if ss.local["choice"] == REF else ss.local["choice"]
        pre = choices.index(want) if want in choices else 0
    ch = st.selectbox("Modèle à expliquer", choices, index=pre)
    if ch.startswith("XGBoost — modèle"):
        b = reference()
        name, model, cfg, thr0 = "XGBoost", b["model"], b["cfg"], b["threshold"]
    else:
        m = ss.models[ch]
        name, model, cfg, thr0 = ch, m["model"], m["cfg"], 0.5
    feats, cols = cfg["features"], cfg["cols"]
    st.caption(f"Modèle actif : {name} · seuil de décision : {thr0:.2f}")

    # ----------------------------------------------------------- Globale
    st.header(f"Analyse globale — {ch}")
    if st.button("Calculer l'explication globale", type="primary"):
        with st.spinner("Calcul des importances et des valeurs SHAP…"):
            Xs = cfg["Xte"].sample(min(300, len(cfg["Xte"])), random_state=42)
            sv, base = core.shap_values(name, model, Xs, cfg["Xtr"])
            nat = core.native_importance(name, model, cols)
            ss.xai[ch] = dict(
                nat=None if nat is None else core.group_series(nat, feats).sort_values(),
                perm=core.grouped_permutation(model, cfg["Xte"], cfg["yte"], cols, feats).sort_values("importance_mean"),
                g=core.group_values(sv, cols, feats), raw=cfg["raw_te"].loc[Xs.index])
    X = ss.xai.get(ch)
    if X:
        c1, c2 = st.columns(2)
        if X["nat"] is not None:
            c1.plotly_chart(px.bar(X["nat"], orientation="h", color=X["nat"], color_continuous_scale="Blues",
                                   template="plotly_dark", labels={"value": "importance", "index": "Variable"},
                                   title=f"Importance native groupée — {name}").update_layout(coloraxis_showscale=False),
                            use_container_width=True)
        else:
            c1.info("KNN n'a pas d'importance native : voir permutation et SHAP.")
        P = X["perm"]
        c2.plotly_chart(px.bar(P, x="importance_mean", y=P.index, orientation="h", error_x="importance_std",
                               color="importance_mean", color_continuous_scale="Blues", template="plotly_dark",
                               labels={"y": "Variable"}, title="Importance par permutation — Average Precision"
                               ).update_layout(coloraxis_showscale=False), use_container_width=True)
        ma = X["g"].abs().mean().sort_values()
        st.plotly_chart(px.bar(ma, orientation="h", color=ma, color_continuous_scale="Blues", template="plotly_dark",
                               labels={"value": "mean_abs_shap", "index": "Variable"},
                               title="Importance SHAP moyenne absolue").update_layout(coloraxis_showscale=False),
                        use_container_width=True)
        st.write(f"Permutation : {P['importance_mean'].idxmax()} arrive en tête ({P['importance_mean'].max():.4f}) · "
                 f"SHAP moyen absolu : {ma.idxmax()} ({ma.max():.4f}).")
        st.subheader("Résumé SHAP groupé — variables originales")
        data = X["raw"].apply(lambda s: s if pd.api.types.is_numeric_dtype(s) else s.astype("category").cat.codes)
        ex = shap.Explanation(values=X["g"].values, data=data.values, feature_names=list(X["g"].columns))
        plt.figure()
        shap.plots.beeswarm(ex, show=False, max_display=12)
        st.pyplot(plt.gcf())
        plt.close()
        cA, cB = st.columns(2)
        vx = cA.selectbox("Variable sur l'axe x", feats, key="dx")
        vc = cB.selectbox("Variable de couleur", [f for f in feats if f != vx] or feats, key="dc")
        dd = pd.DataFrame({vx: X["raw"][vx].values, "Valeur SHAP": X["g"][vx].values, vc: X["raw"][vc].values})
        st.plotly_chart(px.scatter(dd, x=vx, y="Valeur SHAP", color=vc, template="plotly_dark",
                                   color_continuous_scale="RdBu_r", title=f"Dépendance SHAP — {vx}"),
                        use_container_width=True)

    # ----------------------------------------------------------- Locale
    st.header("Explication locale — un individu")
    mode = st.radio("Individu à expliquer", ["Individu du jeu de test", "Individu saisi (page Prédiction)"],
                    index=1 if ss.local else 0, horizontal=True)
    x = raw = None
    if mode.startswith("Individu du jeu"):
        order = cfg["yte"].sort_values(ascending=False).index
        i = st.selectbox("Individu", order, format_func=lambda k: f"#{k} · réel : {POS if cfg['yte'][k] else NEG}")
        x, raw = cfg["Xte"].loc[[i]], cfg["raw_te"].loc[i]
    elif ss.local and set(ss.local["feats"]) == set(feats):
        raw = pd.Series(ss.local["rec"])[feats]
        x = core.encode(pd.DataFrame([ss.local["rec"]])[feats], cols)
    else:
        st.info("Saisissez un individu dans la page Prédiction (même modèle) puis cliquez sur « Expliquer cette prédiction ».")
    if x is not None:
        thr = st.slider("Seuil de décision", 0.0, 1.0, float(thr0), 0.01, key="lthr")
        p = float(core.proba(model, x)[0])
        c = st.columns(3)
        metric(c[0], "Probabilité prédite", f"{p:.3f}")
        metric(c[1], "Seuil", f"{thr:.2f}")
        metric(c[2], "Décision", POS.capitalize() if p >= thr else NEG.capitalize())
        sv, base = core.shap_values(name, model, x, cfg["Xtr"])
        g = core.group_values(sv, cols, feats).iloc[0]
        st.success(core.sentence(g, p, thr, POS))
        o = g.abs().sort_values().index
        exw = shap.Explanation(values=g[o].values, base_values=base,
                               feature_names=[f"{f} = {raw[f]}" for f in o])
        L, R = st.columns(2)
        with L:
            st.subheader("SHAP — waterfall")
            plt.figure()
            shap.plots.waterfall(exw, show=False)
            st.pyplot(plt.gcf())
            plt.close()
        with R:
            st.subheader("LIME — même individu")
            with st.spinner("Calcul LIME…"):
                lime = LimeTabularExplainer(cfg["Xtr"].values, feature_names=cols, class_names=[NEG, POS],
                                            mode="classification", random_state=42)
                e = lime.explain_instance(x.values[0], lambda a: model.predict_proba(pd.DataFrame(a, columns=cols)),
                                          num_features=len(cols), num_samples=2000)
                lw = core.group_series(pd.Series({cols[j]: w for j, w in e.as_map()[1]}), feats)
            lw = lw.sort_values()
            st.plotly_chart(px.bar(lw, orientation="h", template="plotly_dark", labels={"value": "poids LIME", "index": ""},
                                   color=lw > 0, color_discrete_map={True: "#d62728", False: "#4c8bbd"}
                                   ).update_layout(showlegend=False, height=400), use_container_width=True)
        cmp = pd.DataFrame({"SHAP": g, "LIME": lw}).dropna()
        cmp["Même sens"] = np.sign(cmp["SHAP"]) == np.sign(cmp["LIME"])
        top = lambda s: set(s.abs().sort_values(ascending=False).index[:3])
        st.subheader("Comparaison SHAP / LIME")
        st.dataframe(cmp.round(4), use_container_width=True)
        st.write(f"Accord de signe : {cmp['Même sens'].mean():.0%} · variables communes dans le top 3 : "
                 f"{len(top(cmp['SHAP']) & top(cmp['LIME']))}/3. "
                 "(SHAP est exprimé en log-odds pour XGBoost, LIME en probabilité : on compare le signe et le rang, pas l'amplitude.)")
