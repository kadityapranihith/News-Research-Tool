
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain


llm=ChatGoogleGenerativeAI(
            model="gemini-1.5-pro",  # or gemini-1.5-pro
            temperature=0.5,
            google_api_key="AIzaSyBJIuu49S76KqdDslnTnO5GPtGMnJpEcLU"
)
st.title('News Research Tool')
st.sidebar.text("News Articles URLs")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
urls=[]
for i in range(3):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Process URLs")
main_placeholder=st.empty()
if process_url_clicked:
    #load data
    loader=UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data loading.. Started..")
    data=loader.load()
    #split data
    splitter=RecursiveCharacterTextSplitter(
        separators=['\n\n','\n',' '],
        chunk_size=1000,
        chunk_overlap=200
    )

    docs=splitter.split_documents(data)
    main_placeholder.text("Text spiltted...")
    #create embedding
    embedding_moddel=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    st.session_state.vectorstore = FAISS.from_documents(docs, embedding=embedding_moddel)
    main_placeholder.text("Created Embedding...")
    time.sleep(2)
question=main_placeholder.text_input('Question:')
if question:
    if st.session_state.vectorstore is not None:
        chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=st.session_state.vectorstore.as_retriever())
        response=chain({'question':question},return_only_outputs=True)
        st.header("Answer")
        st.write(response['answer'])




