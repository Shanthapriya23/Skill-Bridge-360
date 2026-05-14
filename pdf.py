import os
import logging
import PyPDF2
import chromadb
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from deep_translator import GoogleTranslator
from langdetect import detect


# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Configure Gemini API
genai.configure(api_key="")
model = genai.GenerativeModel("gemini-1.5-flash")

# Initialize ChromaDB client
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("pdf_embeddings")

# Initialize Sentence Transformer for embeddings with fallback
def initialize_embedder():
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        logging.warning(f"Error loading model: {e}")
        return SentenceTransformer('paraphrase-MiniLM-L6-v2')

embedder = initialize_embedder()

# Function to extract text from PDF
def extract_text_from_pdf(file_path):
    try:
        reader = PyPDF2.PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():
            logging.error("No text found in the PDF")
        return text
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to chunk text for RAG
def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# Process PDF and store embeddings
def process_pdf(file_path):
    text = extract_text_from_pdf(file_path)
    if text:
        chunks = chunk_text(text)
        for idx, chunk in enumerate(chunks):
            embedding = embedder.encode(chunk).tolist()
            collection.add(
                embeddings=[embedding],
                metadatas=[{"chunk_id": idx}],
                ids=[str(idx)]
            )
        logging.info("PDF processed and embeddings stored successfully.")
    else:
        logging.error("No text to process from the PDF.")

# Query processing
def query_pdf(query):
    try:
        # Detect the language of the query
        query_lang = detect(query)
        
        # Translate the query to English
        query_en = GoogleTranslator(source="auto", target="en").translate(query)
        
        # Generate the query embedding
        query_embedding = embedder.encode(query_en).tolist()
        
        # Retrieve relevant documents from the collection
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Check if any documents were retrieved
        if not results['documents']:
            logging.warning("No results found for the query.")
            return "No relevant information found in the document."
        
        # Concatenate the retrieved documents to form the context
        context = " ".join([result for sublist in results["documents"] for result in sublist if result])
        
        # Generate a response using the model
        response_en = model.generate_content(f"{context} {query_en}").text
        
        # Translate the response back to the original language
        response_translated = GoogleTranslator(source="en", target=query_lang).translate(response_en)
        
        return response_translated
    except Exception as e:
        logging.error(f"Error processing query: {e}")
        return "An error occurred while processing the query."
