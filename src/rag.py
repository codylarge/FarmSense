import os
from llama_index.core import VectorStoreIndex, Document, StorageContext, load_index_from_storage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from src.keys import load_api_key

# Load API keys
load_api_key("OPENAI_API_KEY")

PERSIST_DIR = "./rag/vector_storage"
CROP_FOLDER = "./rag/crop_texts"  # Folder where crop text files are stored

def loadVectorStorage():
    """
    Load or create vector storage from multiple crop text files.
    """
    if not os.path.exists(CROP_FOLDER):
        print(f"❌ Crop text folder '{CROP_FOLDER}' not found!")
        return

    index = load_and_index_dataset(CROP_FOLDER)

def load_and_index_dataset(folder_path):
    """
    Loads multiple crop-specific text files and creates an indexed vector store.
    """
    try:
        if not os.path.exists(PERSIST_DIR):
            print("🔍 Loading crop documents and building the index...")

            # Read and chunk each crop file
            all_documents = []
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):  # Ensure we only read text files
                    crop_name = file_name.replace(".txt", "")  # Extract crop name
                    file_path = os.path.join(folder_path, file_name)

                    print(f"📂 Processing: {file_name}")

                    # Read the text
                    with open(file_path, "r", encoding="utf-8") as file:
                        text = file.read()

                    # Chunk the text
                    chunked_texts = chunk_text(text)

                    # Convert each chunk into a document with metadata
                    for chunk in chunked_texts:
                        doc = Document(text=chunk, metadata={"crop": crop_name, "source": file_path})

                        all_documents.append(doc)

            # Create vector index
            index = VectorStoreIndex.from_documents(all_documents)

            # Save the indexed vector store
            index.storage_context.persist(persist_dir=PERSIST_DIR)
            print(f"✅ Index created with {len(all_documents)} crop-specific chunks and saved.")
        
        else:
            print("📂 Loading existing index from storage...")
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)  # Load previously saved index
        
        return index
    except Exception as e:
        print(f"❌ Error during indexing or loading: {e}")
        raise

def chunk_text(text, chunk_size=512, overlap=50):
    """
    Chunks text into manageable pieces while preserving meaning.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", " "],  # Prefer logical breaks
        length_function=len
    )
    return splitter.split_text(text)

# Call the function to load or create vector store
loadVectorStorage()
