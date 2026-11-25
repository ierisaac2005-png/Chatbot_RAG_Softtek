"""Interfaz Streamlit. Ejecutar desde la raíz: streamlit run app.py."""
from pathlib import Path
import sys
import streamlit as st
sys.path.insert(0,str(Path(__file__).resolve().parent/'src'))
from chatbot import ChatbotRAG, MODEL_ID
st.set_page_config(page_title='Softtek | Chatbot documental', page_icon='📚', layout='centered')
st.title('Chatbot documental · Softtek')
st.caption('Actividad 5 · Caso académico con 30 documentos simulados')
@st.cache_resource
def cargar_bot():
    return ChatbotRAG()
bot=cargar_bot()
if 'historial' not in st.session_state:st.session_state.historial=[]
if 'activo' not in st.session_state:st.session_state.activo=None
with st.sidebar:
    st.header('Consulta con fuentes')
    st.write('Recuperación TF-IDF + generación local')
    st.caption(MODEL_ID)
    st.write('Ejemplos:')
    st.code('¿De qué trata DOC008?\n¿Qué documentos relacionados existen?\n¿Cómo se restaura un respaldo?')
    st.caption('Primera respuesta: descarga y carga del modelo. No requiere API key.')
    st.write('Documento activo:',st.session_state.activo or 'Ninguno')
    if st.button('Nueva conversación'):
        st.session_state.historial=[];st.session_state.activo=None;st.rerun()
    st.warning('Revisa las fuentes antes de aplicar una recomendación. Las citas no garantizan exactitud.')
for m in st.session_state.historial:
    with st.chat_message(m['role']):
        st.write(m['text'])
        if m.get('fuentes'):
            with st.expander('Documentos utilizados'):
                for f in m['fuentes']:
                    st.markdown(f"**{f['id']} · {f['title']}**")
                    st.write(f['text'])
                    if f['score'] is not None:st.caption(f"Similitud léxica: {f['score']:.4f}")
if pregunta:=st.chat_input('Pregunta sobre la base documental',max_chars=1200):
    st.session_state.historial.append({'role':'user','text':pregunta})
    with st.spinner('Recuperando documentos y generando respuesta…'):
        try:
            r=bot.responder(pregunta,st.session_state.activo)
            st.session_state.activo=r.documento_activo
            st.session_state.historial.append({'role':'assistant','text':r.respuesta,'fuentes':r.fuentes})
        except Exception as exc:
            st.session_state.historial.append({'role':'assistant','text':
                f'No se pudo ejecutar el modelo ({type(exc).__name__}). Revisa conexión y dependencias; no se ha generado una respuesta.'})
    st.rerun()
