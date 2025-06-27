import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.embeddings import HuggingFaceEmbeddings
import time

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain

st.title('News Research Tool')

# --- API KEY ---
st.session_state.api_key = st.sidebar.text_input("Enter your Gemini API key", type="password")

llm = None
if st.session_state.api_key:
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.5,
            google_api_key=st.session_state.api_key
        )
        # Validate the key with a test query
        _ = llm.invoke("Hello")
        st.success("✅ Gemini API key validated!")
    except Exception as e:
        st.error("❌ Invalid or expired Gemini API key. Please check and try again.")
        st.stop()
else:
    st.warning("Please enter your Gemini API key.")

# --- URL Inputs ---
st.sidebar.text("News Articles URLs")
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url.strip():
        urls.append(url.strip())

process_url_clicked = st.sidebar.button("Process URLs")
main_placeholder = st.empty()

if process_url_clicked:
    if not urls:
        st.warning("⚠️ Please enter at least one valid URL.")
    else:
        # Load and split data
        loader = UnstructuredURLLoader(urls=urls)
        main_placeholder.text("Data loading... Started...")
        data = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', ' '],
            chunk_size=1000,
            chunk_overlap=200
        )
        docs = splitter.split_documents(data)
        main_placeholder.text("Text split...")

        # Create embedding
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.vectorstore = FAISS.from_documents(docs, embedding=embedding_model)
        main_placeholder.success("✅ Embeddings created and ready for questions!")
        time.sleep(1)

# --- QA Section ---
question = main_placeholder.text_input('Question:')
if question:
    if st.session_state.vectorstore is not None:
        chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.vectorstore.as_retriever()
        )
        response = chain({'question': question}, return_only_outputs=True)
        st.header("Answer")
        st.write(response['answer'])
        st.subheader("Sources")
        st.write(response['sources'])
    else:
        st.warning("⚠️ Please process URLs before asking a question.")
