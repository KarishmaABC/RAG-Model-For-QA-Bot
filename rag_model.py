import os
from pinecone import Pinecone, ServerlessSpec
import pandas as pd

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize Pinecone
pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENVIRONMENT")
)

# Create Pinecone index
index_name = "qa-bot"
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1024,  # Adjust this to your embedding dimension
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',   # Specify the cloud provider if necessary
            region='us-west-1'  # Specify the region
        )
    )

# Connect to the index
index = pc.Index(index_name)

# Function to generate an answer using the index
def generate_answer(query):
    # Add your logic to process the query and get the answer
    # Example: You might want to search the index using the query
    # This is a placeholder and needs to be implemented according to your requirements
    results = index.query(query)  # Replace with actual query handling logic
    answer = results.get("matches", [{}])[0].get("metadata", {}).get("text", "No answer found.")
    return answer

# Function to add documents to the Pinecone index
def add_documents_to_index(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    
    for i, row in df.iterrows():
        metadata = {"title": row["title"], "text": row["text"]}
        index.upsert(vectors=[(f"id-{i}", row["embedding"], metadata)])
    print(f"Added {len(df)} documents to the Pinecone index.")
