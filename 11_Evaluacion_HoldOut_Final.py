#!/usr/bin/env python
# coding: utf-8

# # Fase 5 - Actividad Final: Evaluacion Hold-out sobre el Conjunto Ciego
# 
# Este notebook toma los **modelos finales ya entrenados** y los evalua, por primera y unica vez, contra el conjunto de datos `test_blind_set.csv`. Este conjunto de datos representa el 20% del corpus original y **jamas ha sido expuesto** a ningun proceso de entrenamiento, vectorizacion o ajuste de hiperparametros.
# 
# El objetivo es obtener metricas de generalizacion reales y puras para confirmar la solidez del modelo elegido antes de ponerlo en produccion.

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import time
import warnings

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

warnings.filterwarnings('ignore')
sns.set_theme(style="whitegrid")


# ## 1. Carga del Conjunto de Prueba Ciego (Hold-out)

# In[ ]:


TEST_PATH = 'data/test_blind_set.csv'

df_test = pd.read_csv(TEST_PATH, sep=';')
df_test['texto_lineal'] = df_test['texto_lineal'].fillna('')
df_test['texto_beto'] = df_test['texto_beto'].fillna('')

print(f"Conjunto Ciego (Hold-out) cargado.")
print(f"   Total de registros : {len(df_test):,}")
print(f"   Distribucion de clases:\n{df_test['label'].value_counts(normalize=True) * 100}")

y_true = df_test['label'].values


# ## 2. Helper: Funcion para graficar Matriz de Confusion

# In[ ]:


def graficar_matriz(y_real, y_pred, titulo, ax):
    cm = confusion_matrix(y_real, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Falso (0)', 'Real (1)'],
                yticklabels=['Falso (0)', 'Real (1)'])
    ax.set_title(titulo, fontsize=14)
    ax.set_ylabel('Valor Real')
    ax.set_xlabel('Prediccion')


# ## 3. Evaluacion: Modelos Lineales (SVM y Regresion Logistica)

# In[ ]:


# Rutas a los modelos ganadores de la malla ampliada (NB-09B)
RUTA_SVM = 'data/best_svm_ampliada.joblib'
RUTA_LOG = 'data/best_log_ampliada.joblib'

resultados_lineales = {}

# ---------------------------------------------------------
# SVM (LinearSVC)
# ---------------------------------------------------------
if os.path.exists(RUTA_SVM):
    print("Evaluando SVM (LinearSVC) con malla ampliada...")
    pipeline_svm = joblib.load(RUTA_SVM)

    t_start = time.perf_counter()
    y_pred_svm = pipeline_svm.predict(df_test[['texto_lineal']])
    t_end = time.perf_counter()

    f1_svm = f1_score(y_true, y_pred_svm, average='macro')
    acc_svm = accuracy_score(y_true, y_pred_svm)

    resultados_lineales['SVM'] = {
        'pred': y_pred_svm,
        'f1': f1_svm,
        'acc': acc_svm,
        'time_ms_per_sample': ((t_end - t_start) / len(df_test)) * 1000
    }
    print(f"  [OK] F1-Macro: {f1_svm:.4f} | Accuracy: {acc_svm:.4f}")
else:
    print(f"  [ERROR] No se encontro {RUTA_SVM}")

# ---------------------------------------------------------
# Regresion Logistica
# ---------------------------------------------------------
if os.path.exists(RUTA_LOG):
    print("\nEvaluando Regresion Logistica con malla ampliada...")
    pipeline_log = joblib.load(RUTA_LOG)

    t_start = time.perf_counter()
    y_pred_log = pipeline_log.predict(df_test[['texto_lineal']])
    t_end = time.perf_counter()

    f1_log = f1_score(y_true, y_pred_log, average='macro')
    acc_log = accuracy_score(y_true, y_pred_log)

    resultados_lineales['LogReg'] = {
        'pred': y_pred_log,
        'f1': f1_log,
        'acc': acc_log,
        'time_ms_per_sample': ((t_end - t_start) / len(df_test)) * 1000
    }
    print(f"  [OK] F1-Macro: {f1_log:.4f} | Accuracy: {acc_log:.4f}")
else:
    print(f"  [ERROR] No se encontro {RUTA_LOG}")


# ## 4. Evaluacion: Transformer (BETO)
# 
# Esta evaluacion requiere PyTorch y HuggingFace. Si la carpeta del modelo BETO no se encuentra o el entorno no tiene las dependencias, se saltara esta prueba.

# In[ ]:


RUTA_BETO = 'beto_final_model'
resultados_beto = None

if os.path.exists(RUTA_BETO):
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        from torch.utils.data import DataLoader, Dataset
        from tqdm.auto import tqdm

        print("Cargando modelo BETO y dependencias de PyTorch...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        tokenizer = AutoTokenizer.from_pretrained(RUTA_BETO)
        model = AutoModelForSequenceClassification.from_pretrained(RUTA_BETO)
        model.to(device)
        model.eval()

        class TextDataset(Dataset):
            def __init__(self, texts, tokenizer):
                self.texts = texts
                self.tokenizer = tokenizer

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, idx):
                return self.texts[idx]

        # En el NB-10B se uso texto_beto y un MAX_LENGTH de 512. 
        # Si tu GPU local tiene poca VRAM, baja el BATCH_SIZE (p.ej., a 4 o 8)
        test_dataset = TextDataset(df_test['texto_beto'].tolist(), tokenizer)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        print(f"Evaluando BETO sobre conjunto ciego (Dispositivo: {device})...")
        y_pred_beto = []

        t_start = time.perf_counter()

        with torch.no_grad():
            for batch_texts in tqdm(test_loader, desc="Inferencia BETO"):
                inputs = tokenizer(
                    batch_texts,
                    padding='max_length',
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                outputs = model(**inputs)
                preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
                y_pred_beto.extend(preds)

        t_end = time.perf_counter()

        f1_beto = f1_score(y_true, y_pred_beto, average='macro')
        acc_beto = accuracy_score(y_true, y_pred_beto)

        resultados_beto = {
            'pred': y_pred_beto,
            'f1': f1_beto,
            'acc': acc_beto,
            'time_ms_per_sample': ((t_end - t_start) / len(df_test)) * 1000
        }
        print(f"  [OK] F1-Macro: {f1_beto:.4f} | Accuracy: {acc_beto:.4f}")

    except ImportError:
        print("  [!] No se pudo importar torch o transformers. Se saltara la prueba de BETO.")
else:
    print(f"  [!] Carpeta '{RUTA_BETO}' no encontrada. Se saltara la prueba de BETO.")


# ## 5. Reporte Final y Comparativa
# 
# A continuacion mostramos la matriz de confusion sobre el conjunto *Hold-out* (el test final verdadero).

# In[ ]:


num_modelos = len(resultados_lineales) + (1 if resultados_beto else 0)

if num_modelos > 0:
    fig, axes = plt.subplots(1, num_modelos, figsize=(5 * num_modelos, 4))
    if num_modelos == 1:
        axes = [axes]

    idx = 0
    resumen_lista = []

    for nombre, res in resultados_lineales.items():
        graficar_matriz(y_true, res['pred'], f"Matriz Confusion: {nombre}", axes[idx])
        resumen_lista.append({
            'Modelo': nombre,
            'F1-Score Macro': round(res['f1'], 4),
            'Accuracy': round(res['acc'], 4),
            'Tiempo / Noticia (ms)': round(res['time_ms_per_sample'], 2)
        })
        idx += 1

    if resultados_beto:
        graficar_matriz(y_true, resultados_beto['pred'], "Matriz Confusion: BETO", axes[idx])
        resumen_lista.append({
            'Modelo': 'BETO',
            'F1-Score Macro': round(resultados_beto['f1'], 4),
            'Accuracy': round(resultados_beto['acc'], 4),
            'Tiempo / Noticia (ms)': round(resultados_beto['time_ms_per_sample'], 2)
        })

    plt.tight_layout()
    plt.show()

    df_resumen = pd.DataFrame(resumen_lista)
    print(df_resumen.set_index('Modelo'))

    # Reporte detallado del mejor modelo lineal (usualmente SVM)
    if 'SVM' in resultados_lineales:
        print("\n" + "="*60)
        print(" REPORTE DE CLASIFICACION (HOLD-OUT) - SVM (LinearSVC)")
        print("="*60)
        print(classification_report(y_true, resultados_lineales['SVM']['pred'], target_names=['Falso', 'Real']))
else:
    print("No se encontro ningun modelo para evaluar.")

