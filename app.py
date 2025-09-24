
import os, json
import pandas as pd
import streamlit as st
import joblib

MODEL_FILE = "modelo_c.joblib"
SCHEMA_FILE = "feature_schema_c.json"

@st.cache_resource(show_spinner=False)
def load_schema(schema_path: str = SCHEMA_FILE) -> dict:
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_resource(show_spinner=False)
def load_model(model_path: str = MODEL_FILE):
    return joblib.load(model_path)

def align_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    """Garante que as colunas do CSV batem com o esperado no treino (mesmo nomes e ordem)."""
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    feature_cols = num_cols + cat_cols
    id_cols = [id_vaga_col, id_cand_col]
    needed = feature_cols + id_cols

    # Adicionar colunas faltantes
    for c in needed:
        if c not in df.columns:
            df[c] = pd.NA

    # Remover colunas extras
    df = df[needed]

    # Reordenar exatamente como no treino
    return df[needed]

def rank_candidates(df_pending: pd.DataFrame, schema: dict, model, top_k: int = 10) -> pd.DataFrame:
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    df_aligned = align_columns(df_pending.copy(), schema)
    scores = model.predict_proba(df_aligned[num_cols + cat_cols])[:, 1]
    df_aligned["score"] = scores
    df_aligned["rank"] = df_aligned.groupby(id_vaga_col)["score"].rank(ascending=False, method="first")

    ranking = (
        df_aligned[df_aligned["rank"] <= top_k]
        .sort_values([id_vaga_col, "rank"])
        .reset_index(drop=True)
    )
    return ranking

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Netflix das Vagas — Versão C", layout="wide")
st.title("🎬 Netflix das Vagas — Versão C (autônomo)")

uploaded = st.file_uploader("📂 CSV de pendentes (não classificados)", type=["csv"])
top_k = st.sidebar.number_input("Top K por vaga", min_value=1, max_value=50, value=10, step=1)

if uploaded is not None:
    df_pending = pd.read_csv(uploaded)
    schema = load_schema()
    model = load_model()
    ranking = rank_candidates(df_pending, schema, model, top_k=int(top_k))

    st.success("✅ Ranking gerado!")
    vagas = sorted(ranking[schema["id_vaga_col"]].unique())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)

    top = ranking[ranking[schema["id_vaga_col"]] == vaga_sel].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    cols = st.columns(5)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i % 5]:
            st.markdown(f"### 👤 Candidato {row[schema['id_cand_col']]}")
            st.metric("Score", f"{row['score']:.3f}")
            st.caption(f"Rank: {int(row['rank'])}")

    with st.expander("📊 Tabela completa da vaga"):
        st.dataframe(top, use_container_width=True)
else:
    st.info("⏳ Aguardando upload do CSV de pendentes…")
