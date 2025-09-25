# Strikingly Support Chatbot

This project is a **LangChain-based chatbot** that can answer questions about Strikingly's support center articles.  
It retrieves data from the Strikingly Zendesk API, creates embeddings with FAISS, and uses OpenAI models for Q&A.

---

## 📌 Features
- Fetches Strikingly support articles from the Zendesk API.
- Builds a FAISS vectorstore from article text and URLs.
- Supports both:
  - **Task 1**: `VectorDBQAWithSourcesChain`
  - **Task 2**: `ConversationalRetrievalChain` (humanized answers, with sources).
- Easily deployable and runnable in **Google Colab**.

---

## 🚀 Quick Start (Google Colab)

1. **Open Colab** and clone this repo:

```bash
!git clone https://github.com/QuantLevin/strikingly_test_chatbot.git
%cd strikingly_test_chatbot
````

2. **Install dependencies**:

```bash
%pip install -r requirements.txt
```

3. **Set up OpenAI API Key**
   In Colab, go to:
   **Settings → Secrets → User secrets**
   Add a new key named `OPENAI_API_KEY`.

   Then in the notebook:


4. **Run Task 1 (build embeddings from Strikingly support center)**:

```bash
!python content-chatbot/create_embedding.py -m zendesk -z https://support.strikingly.com/api/v2/help_center/en-us/articles.json
```

5. **Ask a question (Task 1)**:

```bash
!python content-chatbot/ask_question.py "How do I reset my password?"
```

6. **Ask a question (Task 2, Conversational Retrieval)**:

```bash
!python content-chatbot/answer_question.py "How do I add a new section to my website?"
```

---

## 📂 Project Structure

```
strikingly_test_chatbot/
│── content-chatbot/
│   ├── create_embedding.py      # Fetch articles & build FAISS vectorstore
│   ├── ask_question.py          # Task 1: VectorDBQAWithSourcesChain
│   ├── answer_question.py       # Task 2: ConversationalRetrievalChain
│   └── requirements.txt
│── README.md
```

---

## 🔑 Notes

* You **must provide your own OpenAI API key** in Colab.
* The FAISS vectorstore is stored locally in `faiss_store`.
* Task 2 returns **humanized answers** while still showing **source URLs**.

---

## 📧 Contact

Maintainer: **Levin**
GitHub: [QuantLevin](https://github.com/QuantLevin)
