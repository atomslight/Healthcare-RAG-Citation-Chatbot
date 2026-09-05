# Transcendence Health Chatbot
### A Hybrid Search & RAG-Powered Assistant for Holistic Healthcare

## 📖 Overview
The Transcendence Health Chatbot is an intelligent conversational assistant designed to help patients seamlessly navigate holistic healthcare services like acupuncture and chiropractic care. By combining advanced hybrid search with state-of-the-art Large Language Models (LLMs), it instantly answers patient questions, explains treatments, and connects them with the right care without hallucinating unverified medical advice.

## ✨ Features
- **Intelligent Routing:** Automatically classifies user queries to fetch the most relevant information (e.g., booking, insurance, specific treatments).
- **Hybrid Search Engine:** Leverages both Dense (Semantic) and Sparse (BM25) retrieval using Qdrant for highly accurate search results.
- **RAG (Retrieval-Augmented Generation):** Grounds all answers strictly in Transcendence Health's verified knowledge base to prevent hallucinated medical advice.
- **Multi-LLM Support:** Uses Google Gemini as the primary reasoning engine with a Groq (Llama 3) fallback for high availability.
- **Continuous Chat Interface:** A robust CLI chat loop that maintains conversational history for natural, context-aware interactions.

## 🛠 Tech Stack
- **Python:** The core programming language powering the logic and data pipelines.
- **Qdrant:** A high-performance vector database used to store embeddings and execute hybrid search queries (Dense + Sparse).
- **Google GenAI (Gemini 2.5 Flash Lite):** The primary LLM for intent routing and generating conversational responses based on retrieved context.
- **Groq (Llama 3.1 8B):** An ultra-fast inference engine acting as a reliable fallback LLM if the primary model fails.
- **LangChain & HuggingFace Embeddings:** Used to generate vector embeddings (e.g., `intfloat/multilingual-e5-small`) for the semantic search pipeline.
- **Rank-BM25:** A classic sparse retrieval algorithm for exact keyword matching, combined with dense search using Reciprocal Rank Fusion (RRF).
- **python-dotenv:** Securely loads environment variables like API keys from a `.env` file.

## 📋 Prerequisites
Before you begin, ensure you have the following installed and set up:
- **Python:** version 3.13 or higher.
- **Git:** For cloning the repository.
- **API Keys:** You will need valid API keys for:
  - [Google Gemini](https://aistudio.google.com/)
  - [Groq](https://console.groq.com/keys)
  - [Qdrant](https://qdrant.tech/) (URL and API Key)

## 🚀 Local Development (Step-by-Step)

### 1. Clone the Repository
Open your terminal and clone the repository to your local machine:
```bash
git clone <your-repository-url>
cd <repository-directory>
```
*(Replace `<your-repository-url>` and `<repository-directory>` with your actual repository details).*

### 2. Install Dependencies
This project uses `uv` or `pip` to manage dependencies. To install the required packages:
```bash
pip install -r requirements.txt
```
*(If you are using `uv`, you can run `uv sync` based on the `uv.lock` file).*

### 3. Environment Setup
You need to provide your API keys to the application.
1. Create a copy of the example environment file (if available) or create a new one:
```bash
cp .env.example .env
```
*(If `.env.example` does not exist, simply create a file named `.env` in the root directory).*

2. Open the `.env` file in your code editor and add your specific keys. It should look like this:
```env
QDRANT_API_KEY=your_qdrant_api_key_here
QDRANT_URL=your_qdrant_cluster_url_here
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Chatbot
To start the conversational CLI assistant, run the main chat script:
```bash
python final_main.py
```
*(Alternatively, you can run `python main.py` or test search functionality with `python search.py`).*

### 5. Build for Production
This project is currently a CLI-based Python application. For production, you would typically containerize this application using Docker or deploy it as an API backend using a framework like FastAPI. Currently, no explicit build step is required beyond installing dependencies and running the script.

## 🧠 How It Works (Architecture)
1. **Ingestion & Embedding:** The system reads medical service data from `transcendence_services.json`. It converts this text into dense vector embeddings (capturing meaning) and sparse tokens (capturing exact keywords) and stores them in Qdrant.
2. **Intent Routing:** When a user asks a question, an initial layer (Python rules + Gemini LLM) classifies the "route" (e.g., is the user asking about insurance? acupuncture? booking?).
3. **Hybrid Search:** Based on the route, Qdrant executes a hybrid search. It finds the most relevant documents by combining Semantic Search and exact Keyword Search (BM25), scoring them using Reciprocal Rank Fusion (RRF).
4. **Generation (RAG):** The top retrieved documents are injected into a strict prompt. The LLM (Gemini or Groq) reads these verified documents and generates a helpful, conversational answer for the user without making up external medical facts.

## 📁 Folder Structure
- `final_main.py`: The entry point for the main conversational chatbot loop and RAG pipeline.
- `search.py`: Contains the logic for the intent router, query classification, and hybrid search execution against Qdrant.
- `ingestion.py` / `ingestion_embedding.py`: Scripts responsible for reading the raw JSON data, generating embeddings, and populating the Qdrant database.
- `transcendence_services.json`: The source of truth knowledge base containing all service, pricing, and contact information.
- `.env`: (Ignored in version control) Stores your private API keys and database URLs.
- `pyproject.toml` / `uv.lock`: Configuration files defining the project metadata and Python dependencies.
