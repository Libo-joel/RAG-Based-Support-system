import os
import cohere
from pinecone import Pinecone, ServerlessSpec
import numpy as np

# Use environment variables for API keys
co = cohere.Client(os.getenv('COHERE_API_KEY'))
pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))

def create_embedding_index(content):
    # Embed the documents
    embeds = co.embed(
        texts=content,
        model='embed-english-v3.0',
        input_type='search_document',
        truncate='END'
    ).embeddings
    
    # Get embedding shape
    shape = np.array(embeds).shape
    print(f"Embeddings shape: {shape}")
    
    # Create Pinecone index
    index_name = 'cohere-pinecone-scraped-data'
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=shape[1],
            metric="cosine",
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
    
    # Get the index
    index = pc.Index(index_name)
    
    # Prepare data for upserting
    batch_size = 128
    ids = [str(i) for i in range(shape[0])]
    meta = [{'text': text} for text in content]
    to_upsert = list(zip(ids, embeds, meta))
    
    # Upsert data in batches
    for i in range(0, shape[0], batch_size):
        i_end = min(i+batch_size, shape[0])
        index.upsert(vectors=to_upsert[i:i_end])
    
    print(index.describe_index_stats())
    
    return index, co

def semantic_search(index, co, query, top_k=5):
    # Embed the query
    xq = co.embed(
        texts=[query],
        model='embed-english-v3.0',
        input_type='search_query',
        truncate='END'
    ).embeddings
    
    print(f"Query embedding shape: {np.array(xq).shape}")
    
    # Query the index and return top matches
    res = index.query(vector=xq, top_k=top_k, include_metadata=True)
    
    # Print matches
    for match in res['matches']:
        print(f"{match['score']:.2f}: {match['metadata']['text']}")
    
    return res

# Example usage
def main():
    # Read content from file
    with open('scraped.txt', 'r', encoding='utf-8') as file:
        content = file.read().splitlines()
    
    # Create index
    index, cohere_client = create_embedding_index(content)
    
    # Perform semantic search
    query = "What protection does the Financial Services Compensation Scheme (FSCS) provide for Moneybox Junior ISA holders?"
    semantic_search(index, cohere_client, query)

if __name__ == "__main__":
    main()
