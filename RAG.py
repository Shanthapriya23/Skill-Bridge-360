import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import io

# Ensure terminal can handle Unicode (Windows-specific fix)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
                            
# Function to initialize models and create the FAISS index
def initialize_rag(documents):
    # Load the embedding model (MiniLM is lightweight)
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    # Generate embeddings for documents
    doc_embeddings = embedding_model.encode(documents)
    
    # Create FAISS index (using L2 distance metric)
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(np.array(doc_embeddings))   # Add embeddings to index
    
    # Load GPT-2 model for generating responses based on retrieved context
    model_name = "gpt2"  # Free GPT-2 model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    return embedding_model, index, tokenizer, model

# RAG function for retrieval and generation
def rag_pipeline(query, documents, embedding_model, index, tokenizer, model):
    query_embedding = embedding_model.encode([query])
    _, I = index.search(np.array(query_embedding), 1)
    retrieved_doc = documents[I[0][0]]  # Retrieve the document based on the index
    input_text = f"Context: {retrieved_doc}\nQuestion: {query}\nAnswer:"
    
    # Tokenize the input and generate a response
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_length=100)
    
    # Decode the response
    decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
    
    # Extract only the answer part
    answer_start = decoded_output.find("Answer:")
    if answer_start == -1:
        answer = decoded_output  # If "Answer:" not found, return full output
    else:
        answer_start += len("Answer:")
        answer = decoded_output[answer_start:].strip()

        # Remove any extra generated "Question:" or "Context:"
        if "Question:" in answer:
            answer = answer.split("Question:")[0].strip()
        if "Context:" in answer:
            answer = answer.split("Context:")[0].strip()

    return answer

# Main function to test the queries
def test_rag():
    # Example documents (replace with actual documents as needed)
    documents = [
        "The Eiffel Tower is in Paris, France. It was constructed in 1887 and is one of the most iconic landmarks in the world.",
        "The capital of Japan is Tokyo. It is the largest city in Japan and one of the most populous cities in the world.",
        "Mistral-7B is an open-source large language model designed for instruction-based tasks. It can be fine-tuned for specific applications.",
        "Artificial intelligence (AI) is intelligence demonstrated by machines, in contrast to the natural intelligence displayed by humans and animals.",
        "The Great Wall of China is a series of fortifications made of various materials, including stone, brick, tamped earth, and wood, built along the northern borders of China.",
        "The capital of the United States is Washington, D.C., not New York City. Washington, D.C. is located on the Potomac River.",
        "Albert Einstein was a theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics.",
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It is used in web development, data science, artificial intelligence, and more.",
        "The Amazon rainforest, also known as the Amazon jungle, is a vast tropical rainforest in South America, known for its biodiversity.",
        "Leonardo da Vinci was an Italian polymath who is widely considered one of the greatest painters of all time. His most famous work is the Mona Lisa."
    ]

    # Initialize the RAG components
    embedding_model, index, tokenizer, model = initialize_rag(documents)
    
    # Test queries
    queries = [
        "When was the Eiffel Tower built?",
        "Where is Tokyo?",
        "What is Mistral-7B used for?",
        "What is artificial intelligence?",
        "Where is the Great Wall of China?",
        "What is the capital of the United States?",
        "Who developed the theory of relativity?",
        "What is Python used for?",
        "What is the Amazon rainforest?",
        "Who painted the Mona Lisa?"
    ]
    
    # Print responses for each query
    for query in queries:
        response = rag_pipeline(query, documents, embedding_model, index, tokenizer, model)
        print(f"🔹 Query: {query}\n🔹 Answer: {response}\n")

# Run the test function
if __name__ == "__main__":
    test_rag()
