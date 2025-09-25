
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
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    feature_cols = num_cols + cat_cols
    id_cols = [id_vaga_col, id_cand_col]
    needed = feature_cols + id_cols

    for c in needed:
        if c not in df.columns:
            df[c] = 0

    df_aligned = df[needed].copy()
    return df_aligned

def rank_candidates(df_pending: pd.DataFrame, schema: dict, model, top_k: int = 10) -> pd.DataFrame:
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    df_aligned = align_columns(df_pending.copy(), schema)
    df_aligned = df_aligned.loc[:, ~df_aligned.columns.duplicated()]

    expected = pd.Index(model.feature_names_in_).drop_duplicates()
    X_input = df_aligned.reindex(columns=expected, fill_value=0)

    scores = model.predict_proba(X_input)[:, 1]
    df_aligned["score"] = pd.Series(scores, index=df_aligned.index).round(2)
    df_aligned["percent_match"] = (df_aligned["score"] * 100).round(1)
    df_aligned["rank"] = df_aligned.groupby(id_vaga_col)["score"].rank(ascending=False, method="first")

    # Garantir colunas obrigatórias
    for col in ["inf_titulo_vaga", "inf_cliente", "perfil_competencias_tecnicas_e_comportamentais",
                "nome", "recrutador", "situacao_candidato"]:
        if col not in df_pending.columns:
            df_aligned[col] = ""

    for col in ["inf_titulo_vaga", "inf_cliente", "perfil_competencias_tecnicas_e_comportamentais",
                "nome", "recrutador", "situacao_candidato"]:
        if col in df_pending.columns:
            df_aligned[col] = df_pending[col]

    out = (
        df_aligned[df_aligned["rank"] <= top_k]
        .sort_values([id_vaga_col, "rank"])
        .reset_index(drop=True)
    )
    return out

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Netflix das Vagas", layout="wide")
st.title("🎬 Netflix das Vagas")

uploaded = st.file_uploader("📂 CSV de pendentes (não classificados)", type=["csv"])
top_k = st.sidebar.number_input("Limite Candidatos / Vaga", min_value=1, max_value=50, value=10, step=1)

if uploaded is not None:
    df_pending = pd.read_csv(uploaded)
    schema = load_schema()
    model = load_model()
    ranking = rank_candidates(df_pending, schema, model, top_k=int(top_k))

    st.success("✅ Ranking gerado!")

    id_vaga_col = schema["id_vaga_col"]
    if "inf_titulo_vaga" in ranking.columns:
        ranking["vaga_display"] = ranking[id_vaga_col].astype(str) + " - " + ranking["inf_titulo_vaga"].astype(str)
    else:
        ranking["vaga_display"] = ranking[id_vaga_col].astype(str)

    vagas = sorted(ranking["vaga_display"].unique())
    vaga_sel = st.sidebar.selectbox("Selecione a vaga", vagas)
    vaga_id = vaga_sel.split(" ")[0]

    top = ranking[ranking[id_vaga_col].astype(str) == vaga_id].sort_values("rank")
    st.subheader(f"Top {len(top)} candidatos para a vaga {vaga_sel}")

    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i % 3]:
            nome = row.get("nome", "")
            id_candidato = row.get(schema["id_cand_col"], "")
            empresa = row.get("inf_cliente", "Empresa não informada")
            st.markdown(f"### 👤 {nome} (ID: {id_candidato})")
            st.caption(f"🏢 {empresa}")
            st.metric("% Match", f"{row['percent_match']:.1f}%")
            st.caption(f"Rank: {int(row['rank'])}")

    # --- Tabela final com colunas exatas ---
    with st.expander("📊 Detalhes completos da vaga e candidatos"):
        cols_show = [
            id_vaga_col, "inf_titulo_vaga", "inf_cliente",
            "perfil_competencias_tecnicas_e_comportamentais",
            schema["id_cand_col"], "nome", "recrutador",
            "situacao_candidato", "percent_match", "rank"
        ]
        cols_show = [c for c in cols_show if c in top.columns]
        table = top[cols_show].copy()

        rename_map = {
            id_vaga_col: "ID Vaga",
            "inf_titulo_vaga": "Nome Vaga",
            "inf_cliente": "Empresa",
            "perfil_competencias_tecnicas_e_comportamentais": "Descrição Vaga",
            schema["id_cand_col"]: "ID Candidato",
            "nome": "Nome Candidato",
            "recrutador": "Recrutador",
            "situacao_candidato": "Status Candidato",
            "percent_match": "% Match",
            "rank": "Rank",
        }
        table = table.rename(columns=rename_map)

        st.dataframe(table, use_container_width=True)

        csv_bytes = table.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 Baixar tabela (CSV)",
            data=csv_bytes,
            file_name="detalhes_vagas_candidatos.csv",
            mime="text/csv",
        )

else:
    st.info("⏳ Aguardando upload do CSV de pendentes…")
