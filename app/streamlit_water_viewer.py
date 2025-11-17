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
import matplotlib.colors as mcolors

# -------------------- IMPORTANTE: Cargar Clases --------------------
# Importa las clases personalizadas para que joblib.load() funcione.
# Asegúrate de que CustomModels.py esté en la misma carpeta.
try:
    from CustomModels import (
        BaseModel,
        CleanPreprocessor,
        QuantileClipper,
        MLPClassifier,
        DecisionTreeSolver,
        KMeansSolver,
        RFClassifier,
        GradientBoostingModel,
        LogisticRegressionModel
    )
    # Agrega aquí cualquier otra clase personalizada que hayas creado
except ImportError:
    st.error("Error: No se pudo encontrar el archivo 'CustomModels.py'. "
             "Asegúrate de que esté en la misma carpeta que este script de Streamlit.")
    st.stop()

# -------------------- Page config --------------------
st.set_page_config(page_title="Water Potability – Model Viewer", layout="wide")
st.title("💧 Water Potability – Model Viewer 💧")
st.caption("Carga paquetes de modelo (.joblib), muestra métricas y predice.")

# -------------------- Settings -----------------------
# El formato model_package ya contiene las métricas, no necesitamos METRICS_DIR
MODELS_DIR = Path(".") # Asumiendo que están en la raíz, o cambia a Path("models")
MODELS_DIR.mkdir(exist_ok=True)


# -------------------- Sidebar ------------------------
st.sidebar.header("Modelos disponibles (.joblib)")
model_paths = sorted(map(Path, glob.glob(str(MODELS_DIR / "*.joblib"))))
if not model_paths:
    st.warning(f"No se encontraron modelos en ./{MODELS_DIR}/*.joblib.")
    st.stop()
    
selected = st.sidebar.selectbox("Selecciona el modelo", options=[p.name for p in model_paths])
uploaded_csv = st.sidebar.file_uploader("Opcional: CSV para re-evaluar (water_potability.csv)", type=["csv"])

# -------------------- Plotting Helpers ------------------------

def plot_cm_from_saved(cm_array, labels=["No Potable (0)", "Potable (1)"]):
    """Plotea una matriz de confusión desde un array de numpy guardado."""
    fig, ax = plt.subplots(figsize=(4, 3))
    im = ax.imshow(cm_array, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm_array.shape[1]), yticks=np.arange(cm_array.shape[0]))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Matriz de Confusión")
    
    thresh = cm_array.max() / 2.
    for i in range(cm_array.shape[0]):
        for j in range(cm_array.shape[1]):
            ax.text(j, i, format(cm_array[i, j], "d"), ha="center", va="center",
                    color="white" if cm_array[i, j] > thresh else "black")
    fig.tight_layout()
    return fig

def plot_roc_from_saved(fpr, tpr, roc_auc):
    """Plotea una curva ROC desde arrays guardados de fpr y tpr."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

# --- Helpers para re-evaluación ---

def plot_new_confusion_matrix(y_true, y_pred):
    """Calcula y plotea una CM para nuevos datos."""
    cm = confusion_matrix(y_true, y_pred)
    return plot_cm_from_saved(cm) # Reutilizamos el helper de ploteo

def plot_new_roc(y_true, y_prob):
    """Calcula y plotea una ROC para nuevos datos."""
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_prob, ax=ax)
    ax.set_title("Curva ROC (Nuevos Datos)")
    fig.tight_layout()
    return fig

# -------------------- Main ---------------------------
if selected:
    path = MODELS_DIR / selected
    try:
        with st.spinner(f"Cargando {selected}..."):
            # 1. Cargar el model_package (diccionario)
            model_package = joblib.load(path)
        
        # 2. Extraer la instancia del modelo
        model_instance = model_package["pipeline"]
        
        # 3. Extraer features y métricas guardadas
        feature_cols = model_package.get("feature_names", [])
        saved_metrics = model_package.get("metrics", {})
        saved_cm = model_package.get("cm")
        saved_fpr = model_package.get("roc_fpr")
        saved_tpr = model_package.get("roc_tpr")

    except Exception as e:
        st.error(f"Error al cargar o interpretar el archivo '{selected}':")
        st.exception(e)
        st.stop()


    # ---------- Predicción de un solo registro (arriba) ----------
    st.markdown("---")
    st.header("🔮 Predicción de un solo registro")

    if not feature_cols:
        st.info("El paquete de modelo no contenía 'feature_names'. "
                "El formulario de predicción no puede mostrarse.")
    else:
        with st.form("predict_form"):
            ui_cols = st.columns(3)
            values = {}
            for i, col in enumerate(feature_cols):
                with ui_cols[i % 3]:
                    # Usamos 0.0 para features numéricas
                    values[col] = st.number_input(col, value=0.0, format="%f")
            submitted = st.form_submit_button("Predecir")

        if submitted:
            try:
                # Crear un DataFrame con las columnas en el orden correcto
                x = pd.DataFrame([values], columns=feature_cols)
                
                # Usar model_instance para predecir
                pred = model_instance.predict(x)[0]
                
                st.success(f"**Resultado: {'Potable (1)' if pred==1 else 'No potable (0)'}**")
                
                # Intentar obtener probabilidades
                try:
                    prob = model_instance.predict_proba(x)
                    # Manejar salida de proba (Keras vs Sklearn)
                    if prob.ndim > 1:
                        prob_potable = float(prob[0, 1])
                    else:
                        prob_potable = float(prob[0])
                        
                    st.info(f"Probabilidad de ser Potable (clase 1): **{prob_potable:.4f}**")
                
                except (AttributeError, NotImplementedError):
                    st.info("Este modelo no implementa 'predict_proba()'.")
                except Exception as prob_e:
                    st.warning(f"No se pudieron obtener probabilidades: {prob_e}")
                    
            except Exception as pred_e:
                st.error("Error durante la predicción:")
                st.exception(pred_e)


    # ---------- Métricas guardadas (del entrenamiento) ----------
    st.markdown("---")
    st.header("📈 Métricas del Set de Test (Guardadas)")
    
    if not saved_metrics:
        st.info("El paquete de modelo no contenía métricas guardadas.")
    else:
        # Mostrar métricas clave
        metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        cols = st.columns(len(metric_keys))
        for i, key in enumerate(metric_keys):
            value = saved_metrics.get(key)
            if value is not None:
                cols[i].metric(key.upper(), f"{value:.4f}")

        # Mostrar plots guardados
        left, right = st.columns(2)
        with left:
            if saved_cm is not None:
                st.pyplot(plot_cm_from_saved(saved_cm), use_container_width=True)
            else:
                st.info("Matriz de confusión no encontrada en el paquete.")
        
        with right:
            if saved_fpr is not None and saved_tpr is not None:
                auc = saved_metrics.get("roc_auc", 0.0)
                st.pyplot(plot_roc_from_saved(saved_fpr, saved_tpr, auc), use_container_width=True)
            else:
                st.info("Datos de curva ROC no encontrados en el paquete.")


    # ---------- Evaluación con CSV (si el usuario lo sube) ----------
    if uploaded_csv is not None:
        st.markdown("---")
        st.header("📊 Re-evaluar con CSV cargado")
        
        try:
            df = pd.read_csv(uploaded_csv)
            df.columns = [c.strip().replace(" ", "_") for c in df.columns]
            assert "Potability" in df.columns, "CSV debe tener la columna 'Potability'."
        except Exception as e:
            st.error(f"No pude leer el CSV: {e}")
            st.stop()

        # Separar X e y
        # Asegurarse de usar solo las columnas que el modelo espera
        X = df[feature_cols] 
        y = df["Potability"].astype(int)
        
        st.markdown(f"**Resultados de la re-evaluación en `{uploaded_csv.name}`**")

        try:
            # Usar model_instance para predecir
            y_pred = model_instance.predict(X)
            
            y_prob = None
            try:
                prob_array = model_instance.predict_proba(X)
                if prob_array.ndim > 1:
                    y_prob = prob_array[:, 1]
                else:
                    y_prob = prob_array
            except (AttributeError, NotImplementedError):
                st.info("El modelo no expone probabilidades; no se puede calcular ROC.")

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
                cols[i % len(cols)].metric(k.upper(), f"{v:.4f}")

            # Plots
            left, right = st.columns(2)
            with left:
                st.markdown("**Matrix de Confusión (Nuevos Datos)**")
                st.pyplot(plot_new_confusion_matrix(y, y_pred), use_container_width=True)
            with right:
                if y_prob is not None:
                    st.markdown("**Curva ROC (Nuevos Datos)**")
                    st.pyplot(plot_new_roc(y, y_prob), use_container_width=True)

        except Exception as eval_e:
            st.error("Error durante la re-evaluación:")
            st.exception(eval_e)


    # ---------- Metadata al final ----------
    st.markdown("---")
    with st.expander("📄 Metadata del paquete del modelo", expanded=False):
        # Mostramos el diccionario del paquete, excepto la instancia del pipeline
        metadata_display = {k: v for k, v in model_package.items() if k != "pipeline"}
        st.json(metadata_display, expanded_keys=["metrics", "feature_names"])
