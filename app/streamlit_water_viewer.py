import json
import glob
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
)
import matplotlib.pyplot as plt

# -------------------- Page config --------------------
st.set_page_config(page_title="Water Potability – Model Viewer", layout="wide")
st.title("💧 Water Potability – Model Viewer 💧")
st.caption("Carga modelos entrenados (.joblib) y predice un registro.")

# -------------------- Settings -----------------------
MODELS_DIR = Path("models")
METRICS_DIR = Path("metrics")
MODELS_DIR.mkdir(exist_ok=True)
METRICS_DIR.mkdir(exist_ok=True)

# -------------------- Sidebar ------------------------
st.sidebar.header("Modelos disponibles (.joblib)")
model_paths = sorted(map(Path, glob.glob(str(MODELS_DIR / "*.joblib"))))
if not model_paths:
    st.warning("No se encontraron modelos en ./models/*.joblib. Entrena y exporta el pipeline primero.")
selected = st.sidebar.selectbox("Selecciona el modelo", options=[p.name for p in model_paths] if model_paths else [])
uploaded_csv = st.sidebar.file_uploader("CSV para evaluar (water_potability.csv)", type=["csv"])

# -------------------- Helpers ------------------------
def load_metrics_for(model_file: Path) -> Dict[str, Any]:
    cand = METRICS_DIR / f"{model_file.stem}.json"
    if cand.exists():
        try:
            return json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def try_infer_feature_names(pipeline, meta: Dict[str, Any]) -> List[str]:
    # 1) Mejor: desde el paso 'preprocess'
    try:
        pre = pipeline.named_steps.get("preprocess")
        if pre is not None and hasattr(pre, "transformers_"):
            for _, _, cols in pre.transformers_:
                if cols is not None:
                    return list(cols)
    except Exception:
        pass
    # 2) Fallback: si el JSON guardó feature_names
    fn = meta.get("feature_names")
    if isinstance(fn, list) and fn:
        return fn
    # 3) Sin columnas
    return []

def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]), yticks=np.arange(cm.shape[0]))
    ax.set_xticklabels(["Not Potable", "Potable"])
    ax.set_yticklabels(["Not Potable", "Potable"])
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_roc(y_true, y_prob):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    return fig

# -------------------- Main ---------------------------
if selected:
    path = MODELS_DIR / selected
    with st.spinner(f"Cargando {selected}..."):
        pipeline = joblib.load(path)
    meta = load_metrics_for(path)

    # ---------- Predicción de un solo registro (arriba) ----------
    st.markdown("---")
    st.header("Predicción de un solo registro")
    #st.caption("Si no cargas CSV, inferimos las columnas del pipeline. Si no es posible, carga un CSV en la barra lateral.")

    feature_cols = try_infer_feature_names(pipeline, meta)
    if not feature_cols:
        st.info("No pude inferir columnas. Si cargas un CSV en la barra lateral, tomaremos los nombres de allí.")
    # Si el usuario subió CSV, úsalo para las columnas por si el pipeline no las expone
    if uploaded_csv is not None and not feature_cols:
        try:
            df_tmp = pd.read_csv(uploaded_csv)
            df_tmp.columns = [c.strip().replace(" ", "_") for c in df_tmp.columns]
            if "Potability" in df_tmp.columns:
                feature_cols = [c for c in df_tmp.columns if c != "Potability"]
        except Exception:
            pass

    # Formulario
    with st.form("predict_form"):
        ui_cols = st.columns(3)
        values = {}
        for i, col in enumerate(feature_cols or []):
            with ui_cols[i % 3]:
                values[col] = st.number_input(col, value=0.0)
        submitted = st.form_submit_button("Predecir")

    if submitted:
        if not feature_cols:
            st.warning("No hay columnas inferidas. Carga un CSV en la barra lateral o exporta un pipeline con 'preprocess'.")
        else:
            x = pd.DataFrame([values])
            pred = pipeline.predict(x)[0]
            st.success(f"Resultado: {'Potable (1)' if pred==1 else 'No potable (0)'}")
            try:
                prob = float(pipeline.predict_proba(x)[:, 1][0])
                st.info(f"Probabilidad clase 1 (potable): {prob:.3f}")
            except Exception:
                pass

    # ---------- Evaluación con CSV (si el usuario lo sube) ----------
    if uploaded_csv is not None:
        st.markdown("---")
        st.header("📊 Resultados en CSV cargado")
        try:
            df = pd.read_csv(uploaded_csv)
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]
            assert "Potability" in df.columns, "CSV debe tener la columna 'Potability' para evaluar."
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            st.stop()

        X = df.drop(columns=["Potability"]); y = df["Potability"].astype(int)
        y_pred = pipeline.predict(X)

        y_prob = None
        clf = getattr(pipeline, "named_steps", {}).get("clf", None)
        if clf is not None and hasattr(clf, "predict_proba"):
            try:
                y_prob = pipeline.predict_proba(X)[:, 1]
            except Exception:
                y_prob = None

        # Métricas
        cols = st.columns(5)
        met = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "precision": float(precision_score(y, y_pred, zero_division=0)),
            "recall": float(recall_score(y, y_pred, zero_division=0)),
            "f1": float(f1_score(y, y_pred, zero_division=0)),
        }
        if y_prob is not None:
            try:
                met["roc_auc"] = float(roc_auc_score(y, y_prob))
            except Exception:
                pass
        for i, (k, v) in enumerate(met.items()):
            cols[i % len(cols)].metric(k.upper(), f"{v:.3f}")

        # Plots
        left, right = st.columns(2)
        with left:
            st.markdown("**Matriz de confusión**")
            st.pyplot(plot_confusion_matrix(y, y_pred), use_container_width=True)
        with right:
            if y_prob is not None:
                st.markdown("**Curva ROC**")
                st.pyplot(plot_roc(y, y_prob), use_container_width=True)
            else:
                st.info("El modelo no expone probabilidades; no se puede trazar ROC.")

    # ---------- Metadata al final ----------
    st.markdown("---")
    with st.expander("📄 Metadata del modelo", expanded=False):
        st.json({
            "model_file": selected,
            "saved_from": meta.get("saved_from", "unknown"),
            "model_name": meta.get("model_name", "unknown"),
            "params": meta.get("params", {}),
            "feature_names": meta.get("feature_names", None),
            "commit": meta.get("commit", None),
            "note": meta.get("note", None),
        })
