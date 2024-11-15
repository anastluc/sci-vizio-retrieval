import base64
import chromadb
from chromadb.utils import embedding_functions

# Connect to ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection(name="image_analysis")

# Query by text
results = collection.query(
    query_texts=["flowchart showing process steps"],
    n_results=5
)

# print(results.keys())

# print(type(results["ids"]))
# print(type(results["distances"]))
# print(type(results["metadatas"]))
# print(type(results["embeddings"]))
# print(type(results["documents"]))
# print(type(results["uris"]))
# print(type(results["data"]))


# print(results["ids"][0:5])
# print(results["distances"][0:5])
# print(results["metadatas"][0][0].keys())
# print(results["documents"][0])

# sorted_dis = sorted(results["distances"][0], reverse=False)
# print(sorted_dis)
# Get documents with their images
for idx, doc in enumerate(results['documents'][0]):
    metadata = results['metadatas'][0][idx]
    print(f"Document ID: {results['ids'][0][idx]}")
    print(f"PDF File: {metadata['pdf_file']}")
    print(f"Image Type: {metadata['image_type']}")
    print(f"Title: {metadata['title']}")
    print("")
    
    # Convert base64 back to image if needed
    # image_data = metadata['image_data']
    # image_bytes = base64.b64decode(image_data)