# Medium Article RAG API 📚🤖

A serverless Retrieval-Augmented Generation (RAG) API built to answer questions based strictly on a curated dataset of Medium articles. This project connects a Pinecone vector database with an LLM via LangChain, strictly enforcing factual grounding and preventing external hallucinations.

**Author:** Ishai Assulin  

## 🏗️ Architecture & Tech Stack
* **Framework:** Serverless Python (Vercel)
* **Orchestration:** LangChain (`langchain-openai`, `langchain-pinecone`)
* **Vector Database:** Pinecone (Serverless, Cosine similarity, 1536 dimensions)
* **LLM & Embeddings:** `4UHRUIN-gpt-5-mini` and `4UHRUIN-text-embedding-3-small` (via LLMod proxy)
* **Observability:** LangSmith

## ✨ Key Features
* **Strict Grounding:** The model is constrained by a system prompt to answer *only* using the retrieved database context. If the answer is missing, it explicitly returns: `"I don't know based on the provided Medium articles data."`
* **Serverless Routing:** Utilizes Vercel's `vercel.json` rewrites to efficiently route all traffic through a single `api/index.py` handler.
* **Mojibake Cleaning:** Includes a pre-processing pipeline to sanitize UTF-8 encoding errors before embedding.

## ⚙️ Hyperparameters
* **Chunk Size:** 512
* **Overlap Ratio:** 0.2 (102 characters)
* **Top K:** 20 

---

## 🚀 API Endpoints

### 1. Retrieve RAG Statistics
**`GET /api/stats`**  
Returns the exact hyperparameters used for chunking and retrieval.

**Response (200 OK):**
```json
{
  "chunk_size": 512,
  "overlap_ratio": 0.2,
  "top_k": 20
}
