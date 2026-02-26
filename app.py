import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


# -------------------------------
# Functions
# -------------------------------

def load_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


def create_llm():
    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )
    return llm


def generate_answer(llm, docs, question):
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build chat history text
    history_text = ""
    for chat in st.session_state.chat_history[-4:]:  # last 4 messages
        role = chat["role"]
        history_text += f"{role}: {chat['content']}\n"

    prompt = ChatPromptTemplate.from_template(
        """
Instructions:
- Answer the question using the provided context and conversation history.
- Give a clear, concise answer (3–5 sentences).
- Do not add unnecessary details or repeat information.
- If the question refers to previous conversation, use the history to understand it.
- If the answer is not present in the context, reply: "Not found in the provided content."


Previous conversation:
{history}

Context:
{context}

Current question:
{question}

Answer based on context and conversation history.
"""
    )

    chain = prompt | llm

    response = chain.invoke({
        "history": history_text,
        "context": context,
        "question": question
    })

    return response.content

# -------------------------------
# Streamlit UI
# -------------------------------



st.title("🔍 AI Research Tool ")


st.caption(
    "First-time setup may take time — content is loaded and converted into embeddings for retrieval."
)




# Initialize session state variables
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "retrieved_docs" not in st.session_state:
    st.session_state.retrieved_docs = None

if "answer" not in st.session_state:
    st.session_state.answer = None

if "show_sources" not in st.session_state:
    st.session_state.show_sources = False

# Chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []




# -------------------------------
# Sidebar - Multiple URLs
# -------------------------------
st.sidebar.title("Data Sources")

url1 = st.sidebar.text_input("URL 1")
url2 = st.sidebar.text_input("URL 2")
url3 = st.sidebar.text_input("URL 3")

process_button = st.sidebar.button("Process URLs")

if process_button:
    urls = [url1, url2, url3]
    urls = [u for u in urls if u.strip() != ""]  # remove empty inputs

    if len(urls) == 0:
        st.sidebar.warning("Please enter at least one URL")
    else:
        with st.spinner("Loading and processing URLs..."):

            all_docs = []

            for url in urls:
                try:
                    docs = load_url(url)
                    all_docs.extend(docs)
                except:
                    st.sidebar.error(f"Failed to load: {url}")

            # Split and create vectorstore
            splits = split_docs(all_docs)
            st.session_state.vectorstore = create_vectorstore(splits)

        st.sidebar.success("URLs processed! You can start asking questions.")


# Display previous chat history
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.write(chat["content"])

# Question section
if st.session_state.vectorstore:

    user_question = st.chat_input("Ask a question")

    if user_question:

        # ---- Show latest user question immediately ----
        with st.chat_message("user"):
            st.write(user_question)

        # Retrieve documents
        retriever = st.session_state.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={
                "k": 4,
                "fetch_k": 20,
                "lambda_mult": 0.5
            }
        )

        docs = retriever.invoke(user_question)
        st.session_state.retrieved_docs = docs

        # Generate answer
        llm = create_llm()
        answer = generate_answer(llm, docs, user_question)

        # ---- Show assistant response ----
        with st.chat_message("assistant"):
            st.write(answer)

        # Save for sources button
        st.session_state.answer = answer
        st.session_state.show_sources = False

        # ---- NOW add both to history ----
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_question
        })

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer
        })

        st.session_state.answer = answer
        st.session_state.show_sources = False




    # Show Answer



        def toggle_sources():
            st.session_state.show_sources = not st.session_state.show_sources


        # Dynamic button text
        button_label = "Hide Sources" if st.session_state.show_sources else "Show Sources"

        st.button(button_label, on_click=toggle_sources)

        # Show sources if enabled
        if st.session_state.show_sources and st.session_state.retrieved_docs:
            st.markdown("### Sources")

            for i, doc in enumerate(st.session_state.retrieved_docs):
                with st.expander(f"Source {i + 1}"):

                    st.write(doc.page_content)
