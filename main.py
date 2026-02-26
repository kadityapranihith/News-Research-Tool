# Latest LangChain RAG with Groq

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()

# -------- Step 1: Load URL --------
def load_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    return docs


# -------- Step 2: Split Text --------
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(docs)


# -------- Step 3: Create Vector DB --------
def create_vectorstore(splits):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(splits, embeddings)
    return vectorstore


# -------- Step 4: Create RAG Chain --------
def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0
    )

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the context below.
        If the answer is not in the context, say "Not found".

        Context:
        {context}

        Question:
        {question}
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
    )

    return rag_chain


# -------- Step 5: Run --------
def main():
    url = input("Enter URL: ")

    print("Loading website...")
    docs = load_url(url)

    print("Splitting...")
    splits = split_docs(docs)

    print("Creating vector database...")
    vectorstore = create_vectorstore(splits)

    rag_chain = create_rag_chain(vectorstore)

    print("\nYou can now ask questions (type 'exit' to stop)\n")

    while True:
        question = input("Question: ")
        if question.lower() == "exit":
            break

        response = rag_chain.invoke(question)
        print("\nAnswer:", response.content, "\n")


if __name__ == "__main__":
    main()
