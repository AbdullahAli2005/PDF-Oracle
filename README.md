# 📄 PDF Oracle

PDF Oracle is a **Streamlit-based application** that lets you **chat with multiple PDFs** using the power of **LangChain** and **Google Generative AI (Gemini)**. Simply upload your PDFs, ask questions, and get accurate, context-aware answers directly from the content of your documents.

---

## 🚀 Features

* 📂 **Upload Multiple PDFs** – Import and work with multiple documents at once.
* 🔎 **Semantic Search** – Finds the most relevant chunks of text for your queries.
* 💬 **Interactive Chat** – Ask natural language questions and get AI-powered answers.
* 🧠 **Powered by LangChain + Gemini** – Combines document embeddings with LLM reasoning.
* ⚡ **Fast & Lightweight** – Simple Streamlit interface, easy to use.

---

## 🛠️ Tech Stack

* [Python 3.10+](https://www.python.org/)
* [Streamlit](https://streamlit.io/)
* [LangChain](https://www.langchain.com/)
* [Google Generative AI (Gemini)](https://ai.google.dev/)
* [PyPDF2](https://pypi.org/project/pypdf2/)
* [FAISS](https://github.com/facebookresearch/faiss)

---

## 📥 Installation

Clone the repository and set up your virtual environment:

```bash
# Clone the repo
git clone https://github.com/your-username/pdf_oracle_masterpiece.git
cd pdf_oracle_masterpiece

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # For Linux/Mac
.venv\Scripts\activate      # For Windows

# Install dependencies
pip install -r requirements.txt
```

---

## 🔑 Setup API Key

This project uses **Google Generative AI (Gemini)**. You’ll need an API key:

1. Get your API key from [Google AI Studio](https://ai.google.dev/).
2. Create a `.env` file in the project root and add:

```bash
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Usage

Run the Streamlit app:

```bash
streamlit run app.py
```

Then open the link shown in your terminal (usually `http://localhost:8501`).

---

## 📚 How It Works

1. **PDF Upload** – You upload one or more PDFs.
2. **Text Extraction** – The text is extracted and split into chunks.
3. **Embeddings** – Chunks are converted into vector embeddings using LangChain.
4. **FAISS Vector Store** – Embeddings are stored for fast retrieval.
5. **Querying** – When you ask a question, relevant chunks are retrieved.
6. **AI Response** – Gemini LLM processes retrieved content and answers your query.

---

## 🖼️ Screenshots

*(Add your app screenshots here)*

---

## 📌 Roadmap

* [ ] Add support for more file types (Word, Excel, etc.)
* [ ] Add chat history and export
* [ ] Improve UI/UX with custom themes
* [ ] Deploy to Streamlit Cloud

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed with ❤️ by Abdullah Ali.
