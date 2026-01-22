from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field, EmailStr
from dotenv import load_dotenv
from typing import Optional, Literal

load_dotenv()
class review(BaseModel):
    title: str
    summary: Optional[str]=Field(description="A brief summary of the review")
    rating: int = Field(ge=1, le=5)
    sentiment: Literal['positive', 'negative', 'neutral']
    
model=ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest"
)
model_structured_output=model.with_structured_output(review)
response=model_structured_output.invoke("Give me a review of the movie Inception")
print(response)