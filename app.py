
import os
import io
import asyncio
from typing import List, Tuple, Dict, Any

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# Updated imports - using only stable community packages
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.documents import Document
from langchain_google_genai import (
    ChatGoogleGenerativeAI,
    GoogleGenerativeAIEmbeddings,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings

from htmlTemplates import css, bot_template, user_template, app_header


# --- Fix for "no current event loop" errors when using gRPC async clients ---
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


def file_digest(content: bytes) -> str:
    import hashlib
    return hashlib.sha1(content).hexdigest()[:10]


def prepare_pdfs(uploaded) -> List[dict]:
    prepared = []
    for up in uploaded:
        data = up.read()
        prepared.append({"name": up.name, "bytes": data, "digest": file_digest(data)})
    return prepared


def extract_documents(prepared) -> Tuple[List[Document], int]:
    docs: List[Document] = []
    total_pages = 0
    # count pages
    for item in prepared:
        reader = PdfReader(io.BytesIO(item["bytes"]))
        total_pages += len(reader.pages)

    progress = st.progress(0.0, text="Extracting text from PDFs...")
    seen = 0
    for item in prepared:
        reader = PdfReader(io.BytesIO(item["bytes"]))
        n = len(reader.pages)
        for i, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            text = text.replace("\x00", "").strip()
            if text:
                docs.append(
                    Document(
                        page_content=text,
                        metadata={"source": item["name"], "page": i, "digest": item["digest"]},
                    )
                )
            seen += 1
            progress.progress(seen / max(total_pages, 1), text=f"Reading {item['name']} (page {i}/{n})")
    progress.empty()
    return docs, total_pages


# chunking + vectorstore
def chunk_documents(page_docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(page_docs)


def build_vectorstore(chunked_docs: List[Document]):
    # Ensure event loop exists before creating embeddings
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(documents=chunked_docs, embedding=embeddings)


# Custom memory store
class SimpleMemoryStore:
    def __init__(self):
        self.store: Dict[str, BaseChatMessageHistory] = {}

    def get_history(self, session_id: str) -> BaseChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]


# LLM + chain using the new approach
def build_chain(vectorstore, model_name: str, temperature: float, top_k: int):
    llm = ChatGoogleGenerativeAI(model=model_name, temperature=temperature)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    
    # Contextualize question prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", """Given a chat history and the latest user question which might reference context in the chat history, 
        formulate a standalone question which can be understood without the chat history. 
        Do NOT answer the question, just reformulate it if needed and otherwise return it as is."""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )
    
    # Answer question prompt
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an assistant for question-answering tasks. 
        Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise.
        
        Context: {context}"""),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create question answering chain
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create retrieval chain
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain


def render_sources(source_documents):
    if not source_documents:
        return
    chips = []
    for d in source_documents:
        meta = getattr(d, "metadata", {}) or {}
        src = meta.get("source", "PDF")
        page = meta.get("page", "?")
        chips.append(f'<span class="chip" title="Page {page}">{src} · p.{page}</span>')
    # deduplicate keep order
    seen, uniq = set(), []
    for c in chips:
        if c not in seen:
            uniq.append(c)
            seen.add(c)
    st.markdown(
        f"""
        <div class="sources">
            <div class="sources-title">Sources</div>
            <div class="chips">{''.join(uniq)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_message(role: str, content: str):
    template = user_template if role == "user" else bot_template
    st.write(template.replace("{{MSG}}", content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(
        page_title="PDF Oracle — Chat with Multiple PDFs",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.write(css, unsafe_allow_html=True)
    st.markdown(app_header, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_ready" not in st.session_state:
        st.session_state.vector_ready = False
    if "last_sources" not in st.session_state:
        st.session_state.last_sources = []
    if "memory_store" not in st.session_state:
        st.session_state.memory_store = SimpleMemoryStore()

    with st.sidebar:
        st.subheader("📄 Your documents")
        uploaded = st.file_uploader(
            "Upload one or more PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select multiple at once or add more later.",
        )

        st.markdown("---")
        st.subheader("🧠 Retrieval Settings")
        col_a, col_b = st.columns(2)
        with col_a:
            chunk_size = st.number_input("Chunk size", 256, 4000, 1000, step=50)
        with col_b:
            chunk_overlap = st.number_input("Chunk overlap", 0, 1000, 200, step=25)
        top_k = st.slider("Results per query (k)", 1, 15, 4)

        st.markdown("---")
        st.subheader("🤖 Model Settings")
        model = st.selectbox(
            "Google Gemini model",
            ["gemini-2.5-pro", "gemini-2.5-flash"],
            index=0,
        )
        temperature = st.slider("Creativity (temperature)", 0.0, 1.0, 0.2, 0.05)

        st.markdown("---")
        col1, col2 = st.columns([1, 1])
        with col1:
            process = st.button("⚙️ Process documents", use_container_width=True)
        with col2:
            clear_chat = st.button("🧹 Clear chat", use_container_width=True)

        if clear_chat:
            st.session_state.chat_history = []
            st.session_state.memory_store = SimpleMemoryStore()
            st.session_state.last_sources = []
            st.info("Chat cleared.", icon="ℹ️")

        if process:
            if not uploaded:
                st.error("Please upload at least one PDF before processing.", icon="⚠️")
            else:
                if os.getenv("GOOGLE_API_KEY") in (None, "", "your-key-here"):
                    st.error("Missing GOOGLE_API_KEY. Set it in your Streamlit secrets or .env file.", icon="⚠️")
                else:
                    with st.spinner("Crunching your documents..."):
                        prepared = prepare_pdfs(uploaded)
                        page_docs, total_pages = extract_documents(prepared)
                        if not page_docs:
                            st.error("No extractable text found in the uploaded PDFs.", icon="⚠️")
                        else:
                            chunks = chunk_documents(page_docs, chunk_size, chunk_overlap)
                            vector = build_vectorstore(chunks)
                            chain = build_chain(vector, model, temperature, top_k)

                            st.session_state.conversation = chain
                            st.session_state.vector_ready = True
                            st.success(
                                f"Indexed {len(prepared)} file(s), {total_pages} page(s) → {len(chunks)} chunk(s).",
                                icon="✅",
                            )
                            st.balloons()

    # main app
    st.header("Chat with your PDFs")
    st.caption("Ask questions and cite-backed answers will appear below.")

    user_q = st.text_input(
        "Ask a question about your documents:",
        placeholder="e.g., Summarize section 3 of the research paper and list 3 key findings…",
        label_visibility="collapsed",
    )

    if user_q and st.session_state.conversation:
        with st.spinner("Thinking..."):
            # Get chat history for current session
            chat_history = st.session_state.memory_store.get_history("default")
            
            # Prepare input for the chain
            result = st.session_state.conversation.invoke({
                "input": user_q,
                "chat_history": chat_history.messages
            })
            
            # Add the new messages to history
            chat_history.add_user_message(user_q)
            chat_history.add_ai_message(result["answer"])
            
            # Update session state
            st.session_state.chat_history = chat_history.messages
            st.session_state.last_sources = result.get("context", [])

    elif user_q and not st.session_state.conversation:
        st.info("Upload & process PDFs first (left sidebar).", icon="ℹ️")

    chat_block = st.container()
    with chat_block:
        if st.session_state.chat_history:
            for message in st.session_state.chat_history:
                role = "user" if message.type == "human" else "bot"
                render_message(role, message.content)

            if st.session_state.last_sources:
                render_sources(st.session_state.last_sources)
        else:
            st.markdown(
                """
                <div class="empty">
                    <div class="hint">💡 Tip: Upload multiple PDFs, then ask questions like:</div>
                    <ul class="bullets">
                        <li>"Compare the conclusions of the two papers."</li>
                        <li>"What are the definitions on page 7 of <em>docA.pdf</em>?"</li>
                        <li>"Create a 5-point summary with citations."</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

    c1, c2, c3 = st.columns(3)
    with c1:
        if st.session_state.chat_history:
            st.download_button(
                "⬇️ Export chat (.md)",
                data="\n\n".join(
                    [
                        (f"**You:** {m.content}" if m.type == "human" else f"**Assistant:** {m.content}")
                        for m in st.session_state.chat_history
                    ]
                ).encode("utf-8"),
                file_name="chat_export.md",
                mime="text/markdown",
                use_container_width=True,
            )
    with c2:
        st.caption(" ")
    with c3:
        st.caption(" ")


if __name__ == "__main__":
    main()