# --- Legal AI Local Analysis with Ollama, LangChain and Streamlit (Agent + PDF/DOCX/ODT/TXT/HTTPS + LT + Glossary + Sources + Logging) ---

# Requirements:
# pip install langchain chromadb streamlit pypdf python-docx odfpy langchain-community deep-translator requests
# Run Ollama and pull model: https://ollama.com (e.g., `ollama run mistral`)

import os
import streamlit as st
import requests
from langchain_community.document_loaders import PyPDFLoader, UnstructuredWordDocumentLoader, UnstructuredODTLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.embeddings.ollama import OllamaEmbeddings
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deep_translator import GoogleTranslator
from langchain.schema import Document
import json

# --- UI ---
st.set_page_config(page_title="Teismų praktikos AI agentas", layout="wide")
st.title("📚 Teisės AI agentas su dokumentų analize (lietuvių kalba)")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "message_log" not in st.session_state:
    st.session_state.message_log = []

uploaded_file = st.file_uploader("📄 Įkelkite dokumentą PDF, DOCX, ODT, TXT formatu", type=["pdf", "docx", "odt", "txt"])
document_url = st.text_input("🌐 Arba įklijuokite dokumento nuorodą (https://...)")
user_input_lt = st.chat_input("Užduokite klausimą arba komentuokite lietuviškai...")

# --- Sąvokų žodynas ---
glossary = {
    "servitutas": "teisė naudotis kito asmens nekilnojamuoju turtu ribotais tikslais",
    "ieškinys": "oficialus pareiškimas teismui, kuriuo reikalaujama teisinės apsaugos",
    "procesas": "teisminis nagrinėjimas, kai sprendžiamas ginčas ar baudžiamoji byla",
    "LAT": "Lietuvos Aukščiausiasis Teismas",
    "CK": "Civilinis kodeksas"
}

# --- Dokumento apdorojimas ir bazės kūrimas ---
doc_retriever = None
source_doc_map = {}
loader = None

if uploaded_file or document_url:
    with st.spinner("🔍 Apdorojamas dokumentas ir kuriama paieškos bazė..."):
        if uploaded_file:
            ext = uploaded_file.name.split(".")[-1].lower()
            temp_path = f"temp_doc.{ext}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.read())
        elif document_url:
            ext = document_url.split(".")[-1].lower().split("?")[0]
            temp_path = f"temp_doc.{ext}"
            r = requests.get(document_url)
            with open(temp_path, "wb") as f:
                f.write(r.content)

        if ext == "pdf":
            loader = PyPDFLoader(temp_path)
        elif ext == "docx":
            loader = UnstructuredWordDocumentLoader(temp_path)
        elif ext == "odt":
            loader = UnstructuredODTLoader(temp_path)
        elif ext == "txt":
            loader = TextLoader(temp_path)
        else:
            st.error("Nepalaikomas failo formatas.")

        if loader:
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
            split_docs = splitter.split_documents(docs)

            translated_docs = []
            for i, doc in enumerate(split_docs):
                translated_text = GoogleTranslator(source='auto', target='en').translate(doc.page_content)
                translated_doc = Document(page_content=translated_text, metadata={"source": f"Fragmentas {i + 1}"})
                source_doc_map[translated_text[:50]] = doc.page_content[:300]
                translated_docs.append(translated_doc)

            embeddings = OllamaEmbeddings(model="mistral")
            vectordb = Chroma.from_documents(translated_docs, embedding=embeddings)
            doc_retriever = vectordb.as_retriever(search_kwargs={"k": 4})

# --- Agentas ---
llm = ChatOllama(model="mistral")
retriever = doc_retriever if doc_retriever else None
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm, retriever=retriever, return_source_documents=True
)

# --- Vartotojo užklausa ---
if user_input_lt:
    glossary_hits = {term: explanation for term, explanation in glossary.items() if term in user_input_lt.lower()}
    user_input_en = GoogleTranslator(source='auto', target='en').translate(user_input_lt)

    result = qa_chain({
        "question": user_input_en,
        "chat_history": st.session_state.chat_history
    })

    result_text = result["answer"]
    source_docs = result.get("source_documents", [])
    result_lt = GoogleTranslator(source='auto', target='lt').translate(result_text)

    st.session_state.chat_history.append((user_input_en, result_text))
    st.session_state.message_log.append({"user": user_input_lt, "response": result_lt})

    st.chat_message("🧑 Vartotojas").write(user_input_lt)
    st.chat_message("🤖 Agentas").write(result_lt)

    if glossary_hits:
        with st.expander("📘 Paaiškintos sąvokos"):
            for term, explanation in glossary_hits.items():
                st.markdown(f"**{term}** – {explanation}")

    if source_docs:
        with st.expander("📎 Naudoti šaltiniai iš dokumento"):
            for i, doc in enumerate(source_docs):
                original_text = source_doc_map.get(doc.page_content[:50], "(vertimas)")
                st.markdown(f"**Šaltinis {i + 1}:** {doc.metadata.get('source', '')}")
                st.code(original_text)

# --- Pokalbio istorijos išsaugojimas ---
if st.sidebar.button("💾 Atsisiųsti pokalbio istoriją"):
    with open("pokalbis.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.message_log, f, ensure_ascii=False, indent=2)
    with open("pokalbis.json", "rb") as f:
        st.sidebar.download_button("📥 Parsisiųsti JSON", data=f, file_name="pokalbis.json", mime="application/json")

# --- Cleanup ---
for ext in ["pdf", "docx", "odt", "txt"]:
    if os.path.exists(f"temp_doc.{ext}"):
        os.remove(f"temp_doc.{ext}")
