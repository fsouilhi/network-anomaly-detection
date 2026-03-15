"""
Démonstration interactive — Détection d'Anomalies dans les Logs Réseau
Jeu de données : KDD Cup 1999 HTTP | Modèles : Isolation Forest · LOF · Autoencoder PyTorch
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.datasets import fetch_kddcup99
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import (
    confusion_matrix,
    f1_score, roc_auc_score,
    precision_score, recall_score
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Détection d'Anomalies Réseau",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .metric-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #60a5fa;
    }
    .metric-card h3 { margin: 0; font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.05em; }
    .metric-card p  { margin: 0.2rem 0 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
    .alert-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #f87171;
    }
    .alert-card h3 { margin: 0; font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.05em; }
    .alert-card p  { margin: 0.2rem 0 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
    .normal-card {
        background: #1e293b;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        border-left: 3px solid #34d399;
    }
    .normal-card h3 { margin: 0; font-size: 0.8rem; color: #94a3b8; letter-spacing: 0.05em; }
    .normal-card p  { margin: 0.2rem 0 0; font-size: 1.6rem; font-weight: 700; color: #f1f5f9; }
    section[data-testid="stSidebar"] { background: #0f172a; }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] label { color: #f1f5f9 !important; font-weight: 500; }
    section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 { color: #f1f5f9 !important; font-weight: 700; }
    section[data-testid="stSidebar"] [data-baseweb="select"] { background: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] div { color: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] span { color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Modèle Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):

    def __init__(self, dim_entree: int, dim_latent: int = 16):
        super().__init__()
        self.encodeur = nn.Sequential(
            nn.Linear(dim_entree, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, dim_latent), nn.ReLU(),
        )
        self.decodeur = nn.Sequential(
            nn.Linear(dim_latent, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, dim_entree),
        )

    def forward(self, x):
        return self.decodeur(self.encodeur(x))

    def erreur_reconstruction(self, x):
        self.eval()
        with torch.no_grad():
            return ((x - self.forward(x)) ** 2).mean(dim=1).cpu().numpy()


# ---------------------------------------------------------------------------
# Chargement et entraînement (mis en cache)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Chargement des données et entraînement des modèles…")
def charger_et_entrainer():
    try:
        kdd = fetch_kddcup99(subset='http', as_frame=True, percent10=True)
        df = kdd.frame.copy()
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].apply(lambda x: x.decode() if isinstance(x, bytes) else x)
    except Exception:
        NOMS_COLONNES = [
            'duration', 'protocol_type', 'service', 'flag',
            'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
            'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
            'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
            'num_shells', 'num_access_files', 'num_outbound_cmds',
            'is_host_login', 'is_guest_login', 'count', 'srv_count',
            'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
            'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
            'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
            'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate', 'labels', 'difficulty_level'
        ]
        df = pd.read_csv(
            'https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain+.txt',
            names=NOMS_COLONNES
        ).drop(columns=['difficulty_level'])

    features_num = df.select_dtypes(include=np.number).columns.tolist()
    features_cat = [c for c in df.select_dtypes(include='object').columns if c != 'labels']
    features_utiles = [f for f in features_num if df[f].var() > 0]

    if features_cat:
        df_enc = pd.get_dummies(df[features_utiles + features_cat + ['labels']], columns=features_cat)
    else:
        df_enc = df[features_utiles + ['labels']].copy()

    if 'normal.' in df['labels'].values:
        y = (df_enc['labels'] != 'normal.').astype(int).values
    else:
        y = (df_enc['labels'] != 'normal').astype(int).values

    feature_cols = [c for c in df_enc.columns if c != 'labels']
    X = df_enc[feature_cols].values.astype(np.float32)

    X_normal = X[y == 0]
    X_train, X_val = train_test_split(X_normal, test_size=0.15, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_all_s = scaler.transform(X)

    CONTAMINATION = float((y == 1).mean())

    iso = IsolationForest(n_estimators=100, contamination=CONTAMINATION, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)

    lof = LocalOutlierFactor(n_neighbors=20, contamination=CONTAMINATION, novelty=True, n_jobs=-1)
    lof.fit(X_train_s)

    torch.manual_seed(42)
    ae = Autoencoder(X_train_s.shape[1])
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    critere = nn.MSELoss()
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    X_all_t = torch.tensor(X_all_s, dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t), batch_size=512, shuffle=True)

    meilleure, patience, compteur, meilleur_etat = float('inf'), 7, 0, None
    for _ in range(60):
        ae.train()
        for (xb,) in loader:
            opt.zero_grad()
            critere(ae(xb), xb).backward()
            opt.step()
        ae.eval()
        with torch.no_grad():
            pv = critere(ae(X_val_t), X_val_t).item()
        if pv < meilleure:
            meilleure, compteur = pv, 0
            meilleur_etat = {k: v.clone() for k, v in ae.state_dict().items()}
        else:
            compteur += 1
            if compteur >= patience:
                break

    ae.load_state_dict(meilleur_etat)
    erreurs_val = ae.erreur_reconstruction(X_val_t)
    seuil_ae = np.percentile(erreurs_val, 95)

    return iso, lof, ae, scaler, seuil_ae, X_all_s, X_all_t, y, feature_cols, CONTAMINATION


# ---------------------------------------------------------------------------
# Interface
# ---------------------------------------------------------------------------

st.title("Détection d'Anomalies dans les Logs Réseau")
st.caption("KDD Cup 1999 · Isolation Forest · LOF · Autoencoder PyTorch · Approche semi-supervisée")
st.divider()

iso, lof, ae, scaler, seuil_ae, X_all_s, X_all_t, y, feature_cols, CONTAMINATION = charger_et_entrainer()

with st.sidebar:
    st.header("Paramètres")

    st.subheader("Modèle")
    modele_choisi = st.selectbox(
        "Modèle de détection",
        ["Isolation Forest", "Local Outlier Factor", "Autoencoder"]
    )

    st.subheader("Échantillonnage")
    nb_echantillons = st.slider("Nombre d'échantillons", 500, len(y), 2000, step=500)

    if modele_choisi == "Autoencoder":
        st.subheader("Seuil de détection")
        percentile = st.slider("Percentile sur la validation", 80, 99, 95, step=1)
        st.caption(f"Seuil actuel : {seuil_ae:.6f}")

    st.divider()
    lancer = st.button("Lancer l'analyse", type="primary", use_container_width=True)

st.subheader("Aperçu du jeu de données")
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric-card"><h3>ÉCHANTILLONS TOTAUX</h3><p>{len(y):,}</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="normal-card"><h3>TRAFIC NORMAL</h3><p>{(y == 0).sum():,}</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="alert-card"><h3>TRAFIC ANORMAL</h3><p>{(y == 1).sum():,}</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric-card"><h3>TAUX CONTAMINATION</h3><p>{CONTAMINATION:.2%}</p></div>', unsafe_allow_html=True)

st.divider()

if lancer:
    idx = np.random.RandomState(42).choice(len(y), nb_echantillons, replace=False)
    X_sub = X_all_s[idx]
    y_sub = y[idx]

    with st.spinner("Détection en cours…"):
        if modele_choisi == "Isolation Forest":
            y_pred = (iso.predict(X_sub) == -1).astype(int)
            scores = -iso.score_samples(X_sub)
        elif modele_choisi == "Local Outlier Factor":
            y_pred = (lof.predict(X_sub) == -1).astype(int)
            scores = -lof.score_samples(X_sub)
        else:
            X_sub_t = torch.tensor(X_sub, dtype=torch.float32)
            erreurs = ae.erreur_reconstruction(X_sub_t)
            y_pred = (erreurs > seuil_ae).astype(int)
            scores = erreurs

    prec = precision_score(y_sub, y_pred, zero_division=0)
    rap = recall_score(y_sub, y_pred, zero_division=0)
    f1 = f1_score(y_sub, y_pred, zero_division=0)
    auc = roc_auc_score(y_sub, scores)

    st.subheader(f"Résultats — {modele_choisi}")
    d1, d2, d3, d4 = st.columns(4)
    with d1:
        st.markdown(f'<div class="metric-card"><h3>PRÉCISION</h3><p>{prec:.4f}</p></div>', unsafe_allow_html=True)
    with d2:
        st.markdown(f'<div class="normal-card"><h3>RAPPEL</h3><p>{rap:.4f}</p></div>', unsafe_allow_html=True)
    with d3:
        st.markdown(f'<div class="metric-card"><h3>F1-SCORE</h3><p>{f1:.4f}</p></div>', unsafe_allow_html=True)
    with d4:
        st.markdown(f'<div class="metric-card"><h3>AUC-ROC</h3><p>{auc:.4f}</p></div>', unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Matrice de confusion**")
        fig, ax = plt.subplots(figsize=(4, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        mat = confusion_matrix(y_sub, y_pred)
        sns.heatmap(mat, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Normal', 'Attaque'],
                    yticklabels=['Normal', 'Attaque'])
        ax.set_title(modele_choisi, color="#e2e8f0")
        ax.set_ylabel('Étiquette réelle', color="#94a3b8")
        ax.set_xlabel('Étiquette prédite', color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_right:
        st.markdown("**Distribution des scores d'anomalie**")
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")
        for label_val, couleur, nom in zip([0, 1], ['steelblue', 'tomato'], ['Normal', 'Attaque']):
            vals = scores[y_sub == label_val]
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=60, alpha=0.6, label=nom, color=couleur, density=True)
        if modele_choisi == "Autoencoder":
            ax.axvline(seuil_ae, color='#fbbf24', linestyle='--', label=f'Seuil ({seuil_ae:.5f})')
        ax.set_xlabel("Score d'anomalie", color="#94a3b8")
        ax.set_ylabel('Densité', color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        ax.spines[:].set_color("#334155")
        ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown("**Comparaison des trois modèles**")

    with st.spinner("Calcul comparatif…"):
        y_pred_if = (iso.predict(X_sub) == -1).astype(int)
        y_pred_lof = (lof.predict(X_sub) == -1).astype(int)
        X_sub_t = torch.tensor(X_sub, dtype=torch.float32)
        erreurs = ae.erreur_reconstruction(X_sub_t)
        y_pred_ae = (erreurs > seuil_ae).astype(int)

        resultats = {}
        for nom, yp, sc in zip(
            ['Isolation Forest', 'LOF', 'Autoencoder'],
            [y_pred_if, y_pred_lof, y_pred_ae],
            [-iso.score_samples(X_sub), -lof.score_samples(X_sub), erreurs]
        ):
            resultats[nom] = {
                'Précision': precision_score(y_sub, yp, zero_division=0),
                'Rappel': recall_score(y_sub, yp, zero_division=0),
                'F1-score': f1_score(y_sub, yp, zero_division=0),
                'AUC-ROC': roc_auc_score(y_sub, sc),
            }

    df_res = pd.DataFrame(resultats).T.round(4)
    st.dataframe(df_res, use_container_width=True)

    metriques_plot = ['Précision', 'Rappel', 'F1-score', 'AUC-ROC']
    x = np.arange(len(metriques_plot))
    width = 0.25
    couleurs = ['steelblue', 'seagreen', 'tomato']

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f172a")
    ax.set_facecolor("#0f172a")
    for i, (nom, couleur) in enumerate(zip(df_res.index, couleurs)):
        vals = [df_res.loc[nom, m] for m in metriques_plot]
        barres = ax.bar(x + i * width, vals, width, label=nom, color=couleur)
        for b in barres:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f'{b.get_height():.3f}', ha='center', fontsize=7, color="#e2e8f0")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metriques_plot, color="#e2e8f0")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('Score', color="#94a3b8")
    ax.tick_params(colors="#94a3b8")
    ax.spines[:].set_color("#334155")
    ax.legend(fontsize=9, facecolor="#1e293b", edgecolor="#334155", labelcolor="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

else:
    st.info("Configure les paramètres dans la barre latérale et clique sur **Lancer l'analyse**.")
