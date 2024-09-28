import pinecone 
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENVIRONMENT"))

# Create Pinecone index
index_name = "qa-bot"
if index_name not in pinecone.list_indexes():
    pinecone.create_index(index_name, dimension=1024)

# Connect to the index
index = pinecone.Index(index_name)

# Function to add documents to the Pinecone index
def add_documents_to_index(file_path):
    df = pd.read_csv(file_path) if file_path.endswith(".csv") else pd.read_excel(file_path)
    
    for i, row in df.iterrows():
        metadata = {"title": row["title"], "text": row["text"]}
        index.upsert(vectors=[(f"id-{i}", row["embedding"], metadata)])
    print(f"Added {len(df)} documents to the Pinecone index.")

# Run this script separately to populate Pinecone
if __name__ == "__main__":
    add_documents_to_index("documents.csv")
