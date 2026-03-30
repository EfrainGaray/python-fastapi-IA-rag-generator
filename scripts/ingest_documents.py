import os
from elasticsearch import Elasticsearch, helpers
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings, StorageContext, VectorStoreIndex
from llama_index.vector_stores.elasticsearch import ElasticsearchStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from sentence_transformers import SentenceTransformer
import PyPDF2
import pandas as pd
import docx
import sys

# Asegurarse de que la ruta raíz esté en sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from logger import get_logger

# Obtener el logger principal
logger = get_logger()

# Configuración del cliente de Elasticsearch
es = Elasticsearch("http://elasticsearch:9200")
processed_files_path = "/app/processed_files.txt"

# Crear el índice en Elasticsearch si no existe
def create_index():
    index_name = "documents"
    mappings = {
        "mappings": {
            "properties": {
                "content": {"type": "text"},
                "filename": {"type": "keyword"},
                "page_number": {"type": "integer"},
                "embedding": {"type": "dense_vector", "dims": 384}
            }
        }
    }
    if not es.indices.exists(index=index_name):
        es.indices.create(index=index_name, body=mappings)
        logger.info(f"Index '{index_name}' created.")
    else:
        logger.info(f"Index '{index_name}' already exists.")

def read_txt(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

def read_pdf(file_path):
    text_by_page = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text_by_page.append((page.extract_text(), page_num + 1))
    return text_by_page

def read_excel(file_path):
    df = pd.read_excel(file_path)
    text = df.to_string(index=False)
    return [(text, 1)]

def read_docx(file_path):
    doc = docx.Document(file_path)
    text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    return [(text, 1)]

def ingest_documents(directory_path):
    actions = []
    model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para generar embeddings

    if os.path.exists(processed_files_path):
        with open(processed_files_path, "r") as file:
            processed_files = set(file.read().splitlines())
    else:
        processed_files = set()

    new_processed_files = set()

    for filename in os.listdir(directory_path):
        if filename in processed_files:
            logger.info(f"Skipping already processed file: {filename}")
            continue

        file_path = os.path.join(directory_path, filename)
        if filename.endswith(".txt"):
            logger.info(f"Reading text file: {filename}")
            text_by_page = [(read_txt(file_path), 1)]
        elif filename.endswith(".pdf"):
            logger.info(f"Reading PDF file: {filename}")
            text_by_page = read_pdf(file_path)
        elif filename.endswith(".xlsx"):
            logger.info(f"Reading Excel file: {filename}")
            text_by_page = read_excel(file_path)
        elif filename.endswith(".docx"):
            logger.info(f"Reading DOCX file: {filename}")
            text_by_page = read_docx(file_path)
        else:
            logger.warning(f"Unsupported file format: {filename}")
            continue

        parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
        
        for text, page_number in text_by_page:
            chunks = parser.split_text(text)
            logger.info(f"Split {filename} page {page_number} into {len(chunks)} chunks")

            for chunk in chunks:
                embedding = model.encode(chunk, convert_to_tensor=True).tolist()
                actions.append({
                    "_op_type": "index",
                    "_index": "documents",
                    "_source": {
                        "content": chunk,
                        "filename": filename,
                        "page_number": page_number,
                        "embedding": embedding,
                        "metadata": {
                            "source": filename,
                        },
                    },
                })

        new_processed_files.add(filename)

    if actions:
        logger.info(f"Indexing {len(actions)} chunks to Elasticsearch")
        helpers.bulk(es, actions)
        logger.info(f"Indexed {len(actions)} chunks to Elasticsearch successfully")
    else:
        logger.info("No new documents to index")

    with open(processed_files_path, "a") as file:
        for filename in new_processed_files:
            file.write(f"{filename}\n")

if __name__ == "__main__":
    logger.info("Creating index if not exists")
    create_index()

    # Configuración de Embeddings y Settings
    Settings.llm = OpenAI(model="gpt-3.5-turbo")
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-ada-002")
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=20)
    Settings.num_output = 512
    Settings.context_window = 3900

    # Configuración de Vector Store
    vector_store = ElasticsearchStore(
        es_url="http://elasticsearch:9200",
        index_name="documents",
    )

    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    logger.info("Starting document ingestion")
    ingest_documents("/app/docs")
    logger.info("Document ingestion completed")
