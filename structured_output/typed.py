from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict,Optional,Annotated,Literal
import os

load_dotenv()

model=ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest",
)
class review(TypedDict):
    title:str
    summary:Optional[str]
    view:Optional[str]
    rating:Annotated[int,{'ge':1,'le':5}]
    sentiment:Literal['positive','negative','neutral']

model_struchured_output=model.with_structured_output(review)
response=model_struchured_output.invoke("Give me a review of the movie Inception")
print(response)