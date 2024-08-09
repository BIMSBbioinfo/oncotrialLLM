import chromadb
from chromadb.config import Settings
import logging


class ChromaDBHandler:
    def __init__(self, persist_dir, collection_name):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client_settings = None  # Add a new attribute for client_settings
        self.client = self.init_client()
        self.collection = self.load_collection()

    def init_client(self):
        try:
            # Set client_settings.persist_directory if client_settings is provided
            if self.client_settings:
                persist_directory = self.client_settings.persist_directory
            else:
                persist_directory = self.persist_dir

            client = chromadb.Client(Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=persist_directory
            ))
            return client
        except Exception as e:
            logging.error(f"Error initializing ChromaDB client: {e}")
            # Exit the program with a non-zero status code
            exit(1)

    def load_collection(self):
        try:
            # Load the collection with the correct metadata
            metadata = {"hnsw:space": "cosine"}
            collection = self.client.get_or_create_collection(self.collection_name, metadata=metadata)
            return collection
        except Exception as e:
            logging.error(f"Error loading ChromaDB collection: {e}")
            # Exit the program with a non-zero status code
            exit(1)
