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
```

### 2. Query the Dataset
**`POST /api/prompt`**  
Accepts a natural language question, searches the Pinecone vector database, and generates a factually grounded answer.

**Request Body:**
```json
{
  "question": "List exactly 3 articles about education. Return only the titles."
}
```

**Response (200 OK):**
```json
{
  "response": "1. How to keep up with academic literature\n2. How to Write a Good Essay\n3. The Latest: Bloomberg and NYT offer free access to students",
  "context": [
    {
      "article_id": "142",
      "title": "How to keep up with academic literature",
      "chunk": "The actual extracted text from the article...",
      "score": 0.7845
    }
  ],
  "Augmented_prompt": {
    "System": "You are a Medium-article assistant that answers questions strictly...",
    "User": "Retrieved Context: \n... \nQuestion: List exactly 3 articles..."
  }
}
```

---

## 🛠️ Local Setup & Deployment

### Environment Variables
To run this project locally or deploy it to Vercel, you must provide the following environment variables:
* `LLMOD_API_KEY` - University proxy key for OpenAI access
* `PINECONE_API_KEY` - Vector database access
* `LANGSMITH_API_KEY` - (Optional) For tracing and debugging

### Vercel Deployment
This repository is pre-configured for instant Vercel deployment.
1. Connect your GitHub repository to Vercel.
2. Add your Environment Variables in the Vercel dashboard.
3. Deploy! The `vercel.json` file will automatically configure the serverless functions.
