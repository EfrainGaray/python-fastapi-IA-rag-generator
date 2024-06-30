from pydantic import BaseModel

class QuestionRequest(BaseModel):
    question: str
    server: str
    model: str
 