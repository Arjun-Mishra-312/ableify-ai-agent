# Ableify RAG Agent

This project implements a FastAPI backend for an AI agent that uses Retrieval-Augmented Generation (RAG) to provide emotional and psychological support. The agent integrates information from uploaded PDF documents to enhance its responses.

## Project Structure

- **`main.py`**: The main FastAPI application file containing the API endpoints and logic.
- **`requirements.txt`**: Lists the Python dependencies required to run the project.
- **`.env`**: Contains environment variables (e.g., `OPENAI_API_KEY`). **Do not commit this file to version control.**
- **`rag_data/`**: This directory stores the vector embeddings and document chunks used for RAG. It is tracked by Git, allowing you to version control the RAG documents.
- **`.gitignore`**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. **Clone the Repository:**
   ```bash
   git clone <your-repository-url>
   cd Ableify
   ```

2. **Create a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   - Create a `.env` file in the project root.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key_here
     ```

5. **Run the Application:**
   ```bash
   uvicorn main:app --reload
   ```
   The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

- **`/upload-pdf`**: Upload PDF documents for RAG processing.
- **`/chat`**: Interact with the AI agent. The agent uses RAG to provide contextually relevant responses.

## RAG Document Storage

The RAG documents (vector embeddings and document chunks) are stored in the `rag_data/` directory. This directory is tracked by Git, allowing you to version control the RAG documents. Ensure that sensitive information is not included in the uploaded documents.

## Additional Information

- The project uses ChromaDB for vector storage and retrieval.
- The AI agent is designed to provide empathetic and supportive responses, integrating information from uploaded documents naturally without explicitly mentioning the source.
- For more details on the API schema and usage, refer to the API documentation available at `http://127.0.0.1:8000/docs` when the server is running. 