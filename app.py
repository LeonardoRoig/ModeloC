
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

def _first_existing(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None

def align_columns(df: pd.DataFrame, schema: dict) -> pd.DataFrame:
    # Keep only feature + id columns for model inference, but DO NOT lose original df (we'll join metadata later)
    num_cols = schema["num_cols"]
    cat_cols = schema["cat_cols"]
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    feature_cols = num_cols + cat_cols
    id_cols = [id_vaga_col, id_cand_col]
    needed = feature_cols + id_cols

    # Create missing columns with 0 to avoid NaN in model
    for c in needed:
        if c not in df.columns:
            df[c] = 0

    df_aligned = df[needed].copy()
    return df_aligned

def rank_candidates(df_pending: pd.DataFrame, schema: dict, model, top_k: int = 10) -> pd.DataFrame:
    # Resolve metadata columns dynamically (tolerant to different names)
    id_vaga_col = schema["id_vaga_col"]
    id_cand_col = schema["id_cand_col"]

    title_col = _first_existing(df_pending, ["inf_titulo_vaga", "titulo_vaga", "vaga_titulo"])
    empresa_col = _first_existing(df_pending, ["inf_cliente", "empresa", "nome_empresa"])
    qualif_col = _first_existing(df_pending, ["inf_qualificacoes", "qualificacoes", "descricao_vaga"])
    cand_desc_col = _first_existing(df_pending, ["descricao_candidato", "inf_descricao_candidato", "resumo_candidato"])
    cand_nome_col = _first_existing(df_pending, ["nome_candidato", "candidato_nome"])
    recrutador_col = _first_existing(df_pending, ["nome_recrutador", "recrutador_nome", "inf_recrutador"])
    data_insc_col = _first_existing(df_pending, ["data_inscricao", "inscricao_data", "data_candidatura"])

    meta_cols = [c for c in [title_col, empresa_col, qualif_col, cand_desc_col, cand_nome_col, recrutador_col, data_insc_col] if c]

    # Align for model
    df_aligned = align_columns(df_pending.copy(), schema)
    df_aligned = df_aligned.loc[:, ~df_aligned.columns.duplicated()]

    # Expected by model
    expected = pd.Index(model.feature_names_in_).drop_duplicates()
    X_input = df_aligned.reindex(columns=expected, fill_value=0)

    # Predict scores
    scores = model.predict_proba(X_input)[:, 1]
    df_aligned["score"] = pd.Series(scores, index=df_aligned.index).round(2)
    df_aligned["percent_match"] = (df_aligned["score"] * 100).round(1)

    # Rank per vacancy
    df_aligned["rank"] = df_aligned.groupby(id_vaga_col)["score"].rank(ascending=False, method="first")

    # Append metadata (join on index so we don't lose rows)
    if meta_cols:
        df_aligned = df_aligned.join(df_pending[meta_cols])

    # Also keep easy-access canonical names in the result (so UI doesn't need to guess)
    if title_col: df_aligned["inf_titulo_vaga"] = df_aligned[title_col]
    if empresa_col: df_aligned["inf_cliente"] = df_aligned[empresa_col]
    if qualif_col: df_aligned["inf_qualificacoes"] = df_aligned[qualif_col]
    if cand_desc_col: df_aligned["descricao_candidato"] = df_aligned[cand_desc_col]
    if cand_nome_col: df_aligned["nome_candidato"] = df_aligned[cand_nome_col]
    if recrutador_col: df_aligned["nome_recrutador"] = df_aligned[recrutador_col]
    if data_insc_col: df_aligned["data_inscricao"] = df_aligned[data_insc_col]

    # Keep only top_k per vacancy
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

    # Filtro de vagas com ID + Título (usa coluna canônica criada acima)
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
            nome = row.get("nome_candidato", f"Candidato {row[schema['id_cand_col']]}")
            empresa = row.get("inf_cliente", "Empresa não informada")
            st.markdown(f"### 👤 {nome}")
            st.caption(f"🏢 {empresa}")
            st.metric("Match %", f"{row['percent_match']:.1f}%")
            st.caption(f"Rank: {int(row['rank'])}")

    # Tabela final ajustada e renomeada
    with st.expander("📊 Detalhes completos da vaga e candidatos"):
        desired = [
            id_vaga_col, "inf_titulo_vaga", "inf_cliente",
            "inf_qualificacoes", "descricao_candidato",
            schema["id_cand_col"], "nome_candidato",
            "nome_recrutador", "percent_match", "rank"
        ]
        cols_show = [c for c in desired if c in top.columns]
        table = top[cols_show].copy()

        rename_map = {
            id_vaga_col: "ID da Vaga",
            "inf_titulo_vaga": "Título da Vaga",
            "inf_cliente": "Empresa",
            "inf_qualificacoes": "Descrição da Vaga",
            "descricao_candidato": "Descrição do Candidato",
            schema["id_cand_col"]: "ID do Candidato",
            "nome_candidato": "Nome do Candidato",
            "nome_recrutador": "Recrutador",
            "percent_match": "% Match",
            "rank": "Rank",
        }
        table = table.rename(columns=rename_map)

        st.dataframe(table, use_container_width=True)

else:
    st.info("⏳ Aguardando upload do CSV de pendentes…")
