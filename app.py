import pandas as pd, streamlit as st
from train_and_rank_c import rank_candidates

st.set_page_config(page_title='Netflix das Vagas C', layout='wide')
st.title('🎬 Netflix das Vagas — Versão C')

f=st.file_uploader('CSV pendentes',type=['csv'])
if f:
    df=pd.read_csv(f); ranking=rank_candidates(df,10); st.success('Ranking gerado!')
    vagas=sorted(ranking['id_vaga'].unique()); vaga=st.sidebar.selectbox('Selecione a vaga',vagas)
    top=ranking[ranking['id_vaga']==vaga].sort_values('rank')
    cols=st.columns(5)
    for i,(_,r) in enumerate(top.iterrows()):
        with cols[i%5]:
            st.markdown(f"### 👤 Candidato {r['id_candidato']}"); st.metric('Score',f"{r['score']:.3f}"); st.caption(f"Rank: {int(r['rank'])}")
    with st.expander('Tabela completa'):
        st.dataframe(top)
else:
    st.info('Faça upload do CSV de pendentes.')
