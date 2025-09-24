import pandas as pd, joblib, json, numpy as np

SCHEMA_FILE="feature_schema_c.json"; MODEL_FILE="modelo_c.joblib"
def load_schema(): return json.load(open(SCHEMA_FILE))
def load_model(): return joblib.load(MODEL_FILE)
def align_columns(df, schema):
    for c in schema['num_cols']+schema['cat_cols']+[schema['id_vaga_col'],schema['id_cand_col']]:
        if c not in df.columns: df[c]=np.nan
    return df[schema['num_cols']+schema['cat_cols']+[schema['id_vaga_col'],schema['id_cand_col']]]
def rank_candidates(df, top_k=10):
    schema=load_schema(); model=load_model(); df=align_columns(df,schema)
    scores=model.predict_proba(df[schema['num_cols']+schema['cat_cols']])[:,1]
    df['score']=scores; df['rank']=df.groupby(schema['id_vaga_col'])['score'].rank(ascending=False,method='first')
    return df[df['rank']<=top_k].sort_values([schema['id_vaga_col'],'rank']).reset_index(drop=True)
