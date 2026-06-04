import os
import json
from http.server import BaseHTTPRequestHandler
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# hyperparameters
CHUNK_SIZE = 512
OVERLAP_RATIO = 0.2
TOP_K = 20

# env vars
LLMOD_API_KEY = os.environ.get("LLMOD_API_KEY")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

LLMOD_BASE_URL = "https://api.llmod.ai/v1"
PINECONE_INDEX_NAME = "ass1"

# global vars
llm = None
embeddings = None
vectorstore = None

def initialize_components():
    global llm, embeddings, vectorstore
    if llm is None and LLMOD_API_KEY and PINECONE_API_KEY:
        # Enforce required Pinecone credential linkage
        os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
        
        # connect LangSmith
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
        os.environ["LANGSMITH_PROJECT"] = "ass1"
        
        # initiate models
        llm = ChatOpenAI(
            model="4UHRUIN-gpt-5-mini",
            api_key=LLMOD_API_KEY,
            base_url=LLMOD_BASE_URL,
        )

        # initiate embedding client
        embeddings = OpenAIEmbeddings(
            model="4UHRUIN-text-embedding-3-small",
            api_key=LLMOD_API_KEY,
            base_url=LLMOD_BASE_URL
        )
        
        # map the existing index structure
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings
        )

# routes handler
class handler(BaseHTTPRequestHandler):
    
    # GET
    def do_GET(self):
        # match incoming path
        if self.path.endswith('/api/stats'):
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            # json output
            stats_payload = {
                "chunk_size": CHUNK_SIZE,
                "overlap_ratio": OVERLAP_RATIO,
                "top_k": TOP_K
            }
            self.wfile.write(json.dumps(stats_payload).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    # POST
    def do_POST(self):
        if self.path.endswith('/api/prompt'):
            initialize_components()
            
            # edge cases
            if not vectorstore or not llm:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": "Application API keys are not initialized in Vercel environment variables."}).encode('utf-8'))
                return

            # read payload length and decode the body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                body = json.loads(post_data.decode('utf-8'))
                question = body.get("question", "")
                
                # retrieve identical matches and scores - RETRIVAL
                docs_and_scores = vectorstore.similarity_search_with_score(question, k=TOP_K)
                
                context_text = ""
                json_context_array = []
                
                for doc, score in docs_and_scores:
                    context_text += f"Title: {doc.metadata.get('title')}\nAuthor: {doc.metadata.get('authors')}\nContent: {doc.page_content}\n\n"
                    
                    # piece together the required structural list format
                    json_context_array.append({
                        "article_id": str(doc.metadata.get("article_id", "Unknown")),
                        "title": str(doc.metadata.get("title", "Unknown Title")),
                        "chunk": str(doc.page_content),
                        "score": float(score)
                    })
                
                # construct the System String - AUGMENTATION
                system_prompt = (
                    """"
                    You are a Medium-article assistant that answers questions strictly and only
                    based on the Medium articles dataset context provided to you (metadata and article passages).
                    You must not use any external knowledge, the open internet, or information that is not
                    explicitly contained in the retrieved context. If the answer cannot be determined from
                    the provided context, respond: "I don't know based on the provided Medium articles data."
                    Always explain your answer using the given context, quoting or paraphrasing the relevant
                    article passage or metadata when helpful.
                    MAKE SUER that if the answer cannot be determined from the provided context, you MUST return
                    EXACTLY "I don't know based on the provided Medium articles data."
                    """
                )
                user_prompt = f"Retrieved Context:\n{context_text}\n\nQuestion: {question}"
                
                # send response to GPT-5-mini - GENERATION
                response = llm.invoke([
                    ("system", system_prompt), 
                    ("user", user_prompt)
                ])
                
                # compile the strict output
                result_payload = {
                    "response": response.content,
                    "context": json_context_array,
                    "Augmented_prompt": {
                        "System": system_prompt,
                        "User": user_prompt
                    }
                }
                
                # Dispatch back over HTTP
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result_payload).encode('utf-8'))
                
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()
