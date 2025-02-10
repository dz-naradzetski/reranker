import os
import logging
import operator
from uuid import UUID
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.cross_encoders import HuggingFaceCrossEncoder



port = int(os.getenv("PORT", "8787"))
max_length=int(os.getenv("MAX_LENGTH", "512"))
model_name=os.getenv("MODEL", "BAAI/bge-reranker-v2-m3")
device=os.getenv("DEVICE", "cuda")

logging.basicConfig(level=logging.INFO, format='%(levelname)s:     %(message)s')
logging.info("port: %d", port)
logging.info("max_length: %d", max_length)
logging.info("model: %s", model_name)
logging.info("device: %s", device)

model_kwargs = {"device": device}
cross_encoder_model = HuggingFaceCrossEncoder(model_name=model_name)


app = FastAPI()

class Document(BaseModel):
    """
    A model representing a document with an ID and text.

    Attributes:
        id Union[int, str, UUID]: The unique ID of the document.
        text (str): The text content of the document.
    """

    id: Union[int, str, UUID]
    text: str

class RequestData(BaseModel):
    """
    A model representing a request to rerank documents based on their
    similarity to a query.

    Attributes:
        query (str): The query string used for comparison.
        documents (List[Document]): A list of Document objects to be
        ranked.

    Methods:
        construct_pairs(): Returns a list of pairs, where each
        pair consists of the query and a document's text.
    """

    query: str
    documents: List[Document]

    def construct_pairs(self):
        """
        Constructs pairs of query and document texts for scoring.

        Returns:
            A list of pairs, where each pair consists of the
            query and a document's text.

        """
        return [[self.query, doc.text] for doc in self.documents]

class ResponseData(BaseModel):
    """
    A model representing the response to a reranking request.

    Attributes:
        id Union[int, str, UUID]: The ID of the ranked document.
        similarity (float): The calculated similarity score between the query
        and the document's text.
    """
    id: Union[int, str, UUID]
    similarity: float

@app.post("/api/v1/rerank")
async def rerank_documents(request: RequestData):
    """
    Ranks a list of documents based on their similarity to a given query using
    sequence classification.

    Args:
        request (RequestData): A RequestData object containing the query and
        a list of Document objects.

    Returns:
        A ResponseData object containing the ranked documents with their
        corresponding similarity scores.
    """

    response = []
    pairs = request.construct_pairs()
    scores = cross_encoder_model.score(pairs).tolist()
    docs_with_scores = list(zip(request.documents, scores))
    result = sorted(docs_with_scores, key=operator.itemgetter(1), reverse=True)
    for doc, score in result:
        response.append({"id": doc.id, "similarity": score})
    return {"data": response}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=port)
