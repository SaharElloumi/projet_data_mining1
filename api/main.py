"""API d'interprétation (FastAPI) — modèle de référence XGBoost, seuil 0,75.
Lancer :  uvicorn api.main:app --reload --port 8000   (docs : http://localhost:8000/docs)"""
import os
import sys
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict, Field

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import core  # noqa: E402

app = FastAPI(title="Predictive Maintenance Explain API", version="1.0.0",
              description="Prédiction de panne machine (AI4I 2020) et explication SHAP.")
STATE = {}


class Machine(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    Type: Literal["L", "M", "H"] = "L"
    air_temperature: float = Field(300.0, alias="Air temperature", description="Kelvin")
    process_temperature: float = Field(310.0, alias="Process temperature", description="Kelvin")
    rotational_speed: float = Field(1500, alias="Rotational speed", description="rpm")
    torque: float = Field(40.0, alias="Torque", description="Nm")
    tool_wear: float = Field(100, alias="Tool wear", description="minutes")


def _row(m: Machine) -> pd.DataFrame:
    r = {"Type": m.Type, "Air temperature": m.air_temperature,
         "Process temperature": m.process_temperature,
         "Rotational speed": m.rotational_speed, "Torque": m.torque, "Tool wear": m.tool_wear}
    return core.encode(pd.DataFrame([r])[core.FEATURES], STATE["cols"])


@app.on_event("startup")
def _load():
    b = core.load_reference()
    STATE.update(model=b["model"], cols=b["cols"], thr=b["threshold"], cfg=b["cfg"])


@app.get("/health", tags=["Service"])
def health():
    return {"status": "ok", "model": "XGBoost", "threshold": STATE["thr"]}


@app.get("/model-info", tags=["Service"])
def model_info():
    return {"model": "XGBoost", "features": core.FEATURES, "target": core.TARGET,
            "threshold": STATE["thr"], "encoded_columns": STATE["cols"]}


@app.post("/predict", tags=["Prédiction"])
def predict(m: Machine):
    p = float(core.proba(STATE["model"], _row(m))[0])
    return {"score": p, "threshold": STATE["thr"],
            "decision": "Panne" if p >= STATE["thr"] else "Pas de panne"}


@app.post("/explain", tags=["Explication"])
def explain(m: Machine):
    X = _row(m)
    p = float(core.proba(STATE["model"], X)[0])
    sv, base = core.shap_values("XGBoost", STATE["model"], X, STATE["cfg"]["Xtr"])
    g = core.group_values(sv, STATE["cols"], core.FEATURES).iloc[0]
    return {"score": p, "threshold": STATE["thr"], "base_value": base,
            "shap_contributions": g.round(4).to_dict(),
            "summary": core.sentence(g, p, STATE["thr"])}
