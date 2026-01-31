from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from typing import Optional,Literal
import os 
from dotenv import load_dotenv

load_dotenv()

model= ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
class review(BaseModel):
    title:str=Field(description="Title of the movie")
    summary:Optional[str]=Field(description="A brief summary of the movie")
    rating:int=Field(ge=1,le=5,description="Rating of the movie")
    sentiment:Literal['positive','negative','neutral']=Field(description="Sentiment of the movie")  

parser=PydanticOutputParser(pydantic_object=review)

prompt=PromptTemplate(
    input_variables=["movie"],
    partial_variables={"format_instructions":parser.get_format_instructions()},
    template="Give me a review of the movie {movie}\n{format_instructions}"
)

chain=prompt | model | parser

response=chain.invoke({"movie":"Border2"})

print(response)