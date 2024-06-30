from fastapi import APIRouter, HTTPException
from models.request_models import QuestionRequest
from services.llm_factory import LLMFactory
from logger import get_logger
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np

logger = get_logger()
router = APIRouter()
es = Elasticsearch("http://elasticsearch:9200")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Modelo para generar embeddings
rerank_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-2-v2')  # Modelo para re-ranking

@router.post("/ask")
def ask(request: QuestionRequest):
    try:
        question = request.question
        server = request.server
        model_name = request.model

        # Generar embedding de la pregunta
        question_embedding = embedding_model.encode(question, convert_to_tensor=True).tolist()

        # Búsqueda en Elasticsearch usando embedding
        search_response = es.search(index="documents", body={
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": question_embedding}
                    }
                }
            },
            "size": 10
        })

        hits = search_response["hits"]["hits"]
        relevant_texts = [hit["_source"]["content"] for hit in hits]
        filenames_and_pages = [(hit["_source"]["filename"], hit["_source"].get("page_number", "N/A")) for hit in hits]

        if not relevant_texts:
            logger.info(f"No relevant texts found for the question: {question}")
            return {
                "answer": "Lo siento, no se encontraron respuestas relevantes a tu pregunta.",
                "sources": []
            }

        # Reranking usando CrossEncoder
        rerank_inputs = [[question, text] for text in relevant_texts]
        rerank_scores = rerank_model.predict(rerank_inputs)

        reranked_results = sorted(zip(rerank_scores, relevant_texts, filenames_and_pages), key=lambda x: x[0], reverse=True)
        top_texts = [result[1] for result in reranked_results[:3]]
        top_sources = [result[2] for result in reranked_results[:3]]

        # Generación de respuesta usando LLM
        llm_service = LLMFactory.get_llm_service(server)
        context = "\n".join(top_texts)

        if top_texts:
            sources_info = "\n".join([f"Archivo: {filename}, Página: {page}" for filename, page in top_sources])
            prompt_template = """
            Actúa como un asistente amigable para personas mayores. Aquí hay información relevante:
            {context}

            Esta información fue extraída de los siguientes archivos y páginas:
            {sources_info}

            Por favor, responde la siguiente pregunta de manera clara y simple, proporcionando instrucciones paso a paso si es necesario.
            
            Pregunta: {question}
            Respuesta (máximo 2000 caracteres):
            """
            prompt = prompt_template.format(context=context, question=question, sources_info=sources_info)
        else:
            prompt = f"""
            Hola, soy un asistente conversacional diseñado para ayudar a las personas con tareas básicas de uso de su teléfono móvil.
            No encontré información relevante en los documentos disponibles para tu pregunta: {question}
            """

        answer = llm_service.generate(model_name, prompt)

        logger.info(f"Question: {question} - Answer: {answer}")
        return {
            "answer": answer,
            "sources": [{"filename": filename, "page": page} for filename, page in top_sources]
        }

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
