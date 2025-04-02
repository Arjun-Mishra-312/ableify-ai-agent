# main.py

import os
import logging
from typing import List, Dict, Any

import uvicorn
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, UploadFile, HTTPException, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from dotenv import load_dotenv
from openai import OpenAI

# --- Configuration & Initialization ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load environment variables (especially OPENAI_API_KEY)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logging.warning("OPENAI_API_KEY not found in .env file. Please create a .env file and add your key.")
    # Consider raising an error or exiting if the key is strictly required to run
    # raise ValueError("OPENAI_API_KEY not found in environment variables.")

# Initialize OpenAI client
openai_client = OpenAI(api_key=openai_api_key)

# Initialize ChromaDB
# Using a persistent directory outside of the git-ignored 'chroma_data' folder
chroma_client = chromadb.PersistentClient(path="./rag_data")

# Use a pre-built embedding function from sentence-transformers
# Make sure the model is downloaded/available
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Get or create the Chroma collection (using the embedding function)
# Using a fixed collection name for simplicity. Could be dynamic per user/session.
collection_name = "user_docs_collection"
logging.info(f"Initializing ChromaDB collection: {collection_name}")
collection = chroma_client.get_or_create_collection(
    name=collection_name,
    embedding_function=sentence_transformer_ef,
    metadata={"hnsw:space": "cosine"} # Use cosine distance
)
logging.info(f"ChromaDB collection '{collection_name}' initialized.")


# --- FastAPI App ---
app = FastAPI(
    title="Ableify RAG Agent",
    description="AI agent with RAG capabilities to support users, processing uploaded PDFs.",
)

# --- Helper Functions ---

def extract_text_from_pdf(file_stream) -> str:
    """Extracts text from a PDF file stream."""
    try:
        reader = PdfReader(file_stream)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n" # Add newline between pages
        logging.info(f"Successfully extracted text from PDF (length: {len(text)} chars).")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {e}")

def chunk_text(text: str, chunk_size: int = 500, chunk_overlap: int = 50) -> List[str]:
    """Splits text into overlapping chunks."""
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= len(text): # Ensure we don't miss the very end if overlap pushes us past
             break
        # Handle the last chunk if it's smaller than overlap size
        if end >= len(text):
            last_chunk = text[start:]
            if last_chunk: # Add if non-empty
                 chunks.append(last_chunk)
            break # Exit loop after handling the last piece

    # Optional: Add simple cleaning/filtering here if needed
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()] # Remove leading/trailing whitespace and empty chunks
    logging.info(f"Chunked text into {len(chunks)} chunks (size: {chunk_size}, overlap: {chunk_overlap}).")
    return chunks

# --- API Endpoints ---

@app.post("/upload-pdf", summary="Upload PDF for RAG")
async def upload_pdf(file: UploadFile = File(..., description="PDF file to be processed and stored.")):
    """
    Endpoint to upload a PDF file.
    The text content is extracted, chunked, embedded, and stored in ChromaDB
    for later retrieval during chat interactions.
    """
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDF files are accepted.")

    logging.info(f"Received PDF file: {file.filename}")
    try:
        # Read file content
        contents = await file.read()

        # Extract text
        # Pass the file stream (BytesIO) directly if reader supports it, or save temporarily
        import io
        file_stream = io.BytesIO(contents)
        text = extract_text_from_pdf(file_stream)
        file_stream.close()

        if not text:
            logging.warning(f"No text extracted from PDF: {file.filename}")
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        # Chunk text
        chunks = chunk_text(text) # Using default chunk size/overlap

        if not chunks:
            logging.warning(f"Text extracted but resulted in zero chunks for PDF: {file.filename}")
            raise HTTPException(status_code=400, detail="Failed to chunk the extracted text.")

        # Generate IDs and store in ChromaDB
        # Using simple incremental IDs based on filename and chunk index
        # Add filename to metadata for potential filtering later
        ids = [f"{file.filename}_chunk_{i}" for i, _ in enumerate(chunks)]
        metadata = [{"source": file.filename} for _ in chunks]

        logging.info(f"Adding {len(chunks)} chunks from '{file.filename}' to ChromaDB collection '{collection_name}'.")
        collection.add(
            documents=chunks,
            metadatas=metadata,
            ids=ids
        )
        logging.info(f"Successfully added chunks from '{file.filename}' to ChromaDB.")

        return JSONResponse(
            status_code=200,
            content={
                "message": f"Successfully processed and stored '{file.filename}'.",
                "chunks_added": len(chunks)
                }
        )

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        logging.error(f"Error processing PDF upload '{file.filename}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error processing PDF: {e}")
    finally:
        # Ensure file handle is closed if applicable (FastAPI handles this with UploadFile)
        await file.close()
        logging.info(f"Closed file handle for: {file.filename}")


# --- Chat Endpoint Data Models ---

class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender ('user' or 'assistant')")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="Conversation history, including the latest user message.")
    # We keep the model and response_format definitions internal for this endpoint
    # If needed, they could be part of the request, but for now, we fix them.

# The JSON schema provided by the user - defining the expected output structure
CHAT_APP_CARDS_SCHEMA = {
    "name": "chat_app_cards",
    "description": "Schema for chat app cards",
    "strict": True, # Note: OpenAI might ignore 'strict' in some versions/models
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["type", "data", "suggested_responses"],
        "properties": {
            "type": {
                "type": "string",
                "enum": ["TextMessage", "DetailCard", "EmailCard", "BannerCard", "SingleActionCard"]
            },
            "data": {
                "anyOf": [
                    # TextMessage Schema
                    {
                        "type": "object",
                        "description": "Provides a supportive text message focusing on the user's strengths and promoting a positive, growth mindset. The chatbot uses a language filter to avoid medical terms, rephrasing them into more inclusive language, and focuses on environmental challenges instead of medical conditions. Before sending, the response passes through a positivity check to ensure it includes at least one strength-focused comment and avoids negative or limiting language. Please limit responses to a maximum of 10 sentences. If the user expresses suicidal thoughts or self-harm intentions, immediately provide the appropriate helpline in BC, Canada.",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "const": "TextMessage"},
                            "text": {"type": "string"}
                        },
                        "required": ["type", "text"]
                    },
                    # DetailCard Schema
                    {
                        "type": "object",
                        "description": "Provides high-level details to the user, focusing on strengths and positive aspects. Use only after you've sent the user at least 1 TextMessage card first. Use this whenever the user asks for a summary or more detailed information. The response is processed through the language filter and positivity check.",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "const": "DetailCard"},
                            "imageDescription": {"type": "string", "description": "A short description (3-5 words) of a very simple thumbnail image that an AI model can generate to represent this card. The image should be inclusive and sensitive to individuals with disabilities. No diagrams or text should be included."},
                            "title": {"type": "string"},
                            "subtitle": {"type": "string"},
                            "description": {"type": "string", "description": "A high-level summary to the user, ideally 1 sentence long, focusing on positive aspects and strengths."}
                        },
                        "required": ["type", "title", "subtitle", "description", "imageDescription"]
                    },
                     # EmailCard Schema
                    {
                        "type": "object",
                        "description": "Facilitates communication or sharing of resources via email.",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "const": "EmailCard"},
                            "toEmail": {"type": "string"},
                            "subject": {"type": "string"},
                            "text": {"type": "string"}
                        },
                        "required": ["type", "toEmail", "subject", "text"]
                    },
                    # BannerCard Schema
                    {
                        "type": "object",
                        "description": "Specifies or highlights a short tip or suggestion to the user, integrating grounding techniques such as square breathing or meditation to help calm the user if they are stressed. The chatbot uses a stress detector to recognize when the user seems stressed or anxious and offers appropriate grounding techniques.",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "const": "BannerCard"},
                            "description": {"type": "string", "description": "A short sentence describing a grounding technique or positive tip."},
                            "label": {"type": "string", "description": "Maximum 4 words that summarize what type of tip it is. For example: 'Grounding Tip', 'Breathing Exercise', 'Positive Affirmation'."}
                        },
                        "required": ["type", "description", "label"]
                    },
                    # SingleActionCard Schema
                    {
                        "type": "object",
                        "description": "Shows some text and a single button a user can press related to the text. Use when the user asks for help accessing resources, support services, or taking positive actions.",
                        "additionalProperties": False,
                        "properties": {
                            "type": {"type": "string", "const": "SingleActionCard"},
                            "title": {"type": "string", "description": "The title of the action the user could take."},
                            "description": {"type": "string", "description": "A short sentence describing the action the user could take."},
                            "actionTitle": {"type": "string", "description": "Very short name of the action a user could take such as 'Access Resources' or 'Get Support'."}
                        },
                        "required": ["type", "actionTitle", "title", "description"]
                    }
                ]
            },
            "suggested_responses": {
                "type": "array",
                "description": "A list of 2 possible follow-up responses the user can tap to continue the conversation. Each should be no more than 4 words, focusing on strengths and positive actions.",
                "items": {"type": "string"},
                "minItems": 2, # Ensuring exactly 2 suggestions might be good
                "maxItems": 2
            }
        }
    }
}


@app.post("/chat", summary="Chat with RAG-enhanced AI Agent")
async def chat_endpoint(request: ChatRequest = Body(...)):
    """
    Handles chat requests. Retrieves relevant context from stored PDFs (via ChromaDB)
    based on the latest user message, augments the prompt for the OpenAI model,
    and returns a structured JSON response according to the defined schema.
    """
    if not openai_client.api_key:
         raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

    if not request.messages:
        raise HTTPException(status_code=400, detail="No messages provided in the request.")

    # Get the latest user message for RAG query
    last_user_message = next((msg.content for msg in reversed(request.messages) if msg.role == 'user'), None)

    if not last_user_message:
        raise HTTPException(status_code=400, detail="No user message found in the conversation history.")

    logging.info(f"Received chat request. Last user message: '{last_user_message[:50]}...'")

    # --- RAG Step: Query ChromaDB ---
    retrieved_context = ""
    try:
        # Query the collection for documents relevant to the last user message
        # Increase n_results if more context is desired
        results = collection.query(
            query_texts=[last_user_message],
            n_results=3 # Retrieve top 3 relevant chunks
            # Optional: Add 'where' clause for filtering by metadata (e.g., user ID, specific source document)
            # where={"source": "some_specific_document.pdf"}
        )

        # Process and format the retrieved documents
        if results and results.get('documents') and results['documents'][0]:
            retrieved_chunks = results['documents'][0]
            retrieved_context = "\n\n---\n\n".join(retrieved_chunks)
            logging.info(f"Retrieved {len(retrieved_chunks)} relevant chunks from ChromaDB.")
            # logging.debug(f"Retrieved context: {retrieved_context}") # Be careful logging potentially sensitive data
        else:
            logging.info("No relevant documents found in ChromaDB for the query.")

    except Exception as e:
        logging.error(f"Error querying ChromaDB: {e}", exc_info=True)
        # Proceed without context, or return an error? Decide based on desired behavior.
        # For now, proceed without context but log the error.
        retrieved_context = "No context could be retrieved due to an error." # Inform the model? Or just omit? Let's omit for now.
        retrieved_context = ""


    # --- Prepare Prompt for OpenAI ---

    # Define the System Prompt - incorporating RAG instructions & persona refinement
    system_prompt = f"""You are Ableify, a compassionate and supportive AI companion. Your purpose is to be like a caring friend, helping users navigate challenges by focusing on their strengths, fostering a positive outlook, and promoting emotional well-being. Engage in warm, empathetic conversation. Listen actively and respond with understanding.

**Your Core Directives:**
1.  **Empathetic & Strength-Focused:** Always respond with warmth and understanding. Frame conversations positively, acknowledging challenges while highlighting the user's resilience and capabilities. Avoid clinical or overly formal language. Use 'I' statements occasionally to foster connection, but keep the focus primarily on the user.
2.  **Inclusive Language:** Use people-first, inclusive language. Rephrase medical terms into descriptions of experiences or environmental challenges (e.g., instead of 'diagnosed with X', say 'living with the experience of X' or 'navigating challenges related to X').
3.  **Natural Integration of Knowledge:** You may have access to relevant information from documents provided by the user (see 'Retrieved Context' below). If this context directly addresses the user's current topic, **seamlessly weave the relevant information into your supportive response as if it's part of your understanding.** **Crucially, NEVER explicitly state "the document says," "I read that," or mention the source in any way.** Act as if this knowledge is simply part of your supportive dialogue. If the context is irrelevant or unavailable, rely solely on the conversation and your core directives.
4.  **Gentle Redirection:** If the user asks questions that are highly technical, overly broad, unrelated to personal well-being, or seem designed to test your knowledge rather than share feelings, gently guide the focus back to them. Use phrases like: "That's an interesting thought, but right now, we are here to talk about you. How are you feeling at this moment?" or "I understand you're curious about that, but my main focus is supporting you. Could you tell me more about what's on your mind?" or "Let's put that aside for now and focus on you. What's coming up for you today?"
5.  **Grounding Techniques:** If you sense the user might be feeling stressed or anxious, gently offer a simple grounding technique (like mindful breathing or noticing surroundings) using a 'BannerCard'. Frame it as a helpful suggestion, e.g., "Sometimes a deep breath can help. Would you like to try a quick breathing exercise?"
6.  **Resource Navigation:** If the user explicitly asks for help finding resources or support services, use the 'SingleActionCard' to provide a clear next step.
7.  **Summarization (Use Sparingly):** If needed for clarity or if requested, provide brief, positive summaries using a 'DetailCard', but only after having a text-based exchange first ('TextMessage').
8.  **Safety First:** If the user expresses clear suicidal thoughts or self-harm intentions, **immediately** provide only this response: 'If you're in crisis, please reach out for help. In BC, Canada, call or text 9-8-8 anytime. Help is available.' Do not generate any other card type in this specific situation.
9.  **Response Format:** ALWAYS respond ONLY with a JSON object conforming exactly to the 'chat_app_cards' schema. Do not add any text before or after the JSON object. Ensure the 'suggested_responses' array contains exactly two brief (max 4 words), positive, action-oriented follow-up prompts relevant to the conversation.

**Retrieved Context (Integrate Naturally and Subtly if Relevant, NEVER Mention Source):**
---
{retrieved_context if retrieved_context else "No specific context retrieved from user documents for this query."}
---
"""

    # Format conversation history for OpenAI API
    api_messages = [{"role": "system", "content": system_prompt}]
    for msg in request.messages:
        api_messages.append({"role": msg.role, "content": msg.content})

    # --- Call OpenAI API ---
    try:
        logging.info(f"Calling OpenAI API (model: gpt-4o-mini) with {len(api_messages)} messages.")
        # Note: response_format with json_schema requires stream=False
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=api_messages,
            response_format={"type": "json_object"}, # Request JSON output, schema enforced via prompt
            temperature=0.7, # Adjust creativity/factuality balance
            # max_tokens= ? # Consider setting if needed, but JSON mode might manage this
            # stream=False # Cannot stream with response_format=json_object
        )

        # Extract the JSON response content
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            assistant_response_json_str = response.choices[0].message.content
            logging.info("Received response from OpenAI API.")
            # Attempt to parse the JSON string returned by the API
            import json
            try:
                assistant_response_obj = json.loads(assistant_response_json_str)
                # Optional: Validate against the Pydantic models derived from the schema if needed
                # ...
                return JSONResponse(content=assistant_response_obj)
            except json.JSONDecodeError as json_err:
                logging.error(f"Failed to parse JSON response from OpenAI: {json_err}")
                logging.error(f"Raw OpenAI response string: {assistant_response_json_str}")
                raise HTTPException(status_code=500, detail="Failed to parse response from AI model.")
        else:
            logging.error("No valid response content received from OpenAI API.")
            raise HTTPException(status_code=500, detail="AI model did not return a valid response.")

    except Exception as e:
        logging.error(f"Error calling OpenAI API: {e}", exc_info=True)
        # Check for specific OpenAI errors if needed (e.g., AuthenticationError, RateLimitError)
        raise HTTPException(status_code=502, detail=f"Error communicating with AI model: {e}")


# --- Run the App (for local development) ---
if __name__ == "__main__":
    print("--- Starting Ableify RAG Agent API ---")
    print("Access endpoints at http://127.0.0.1:8000")
    print("API docs available at http://127.0.0.1:8000/docs")
    print("Make sure to create a '.env' file with your 'OPENAI_API_KEY'.")
    print("Ensure ChromaDB data directory './rag_data' exists or can be created.")
    print("---------------------------------------")
    # Ensure rag_data directory exists
    if not os.path.exists("./rag_data"):
        os.makedirs("./rag_data")
        logging.info("Created './rag_data' directory for persistent vector storage.")

    # Use the PORT environment variable provided by Render, or default to 8000
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)