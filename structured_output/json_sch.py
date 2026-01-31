from langchain_google_genai import ChatGoogleGenerativeAI
from typing import Optional,Annotated,Literal
from dotenv import load_dotenv
import os
load_dotenv()

model=ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

json_schema={
    "type":"object",
    "properties":{
        "summary":{
            "type":"string"
        },
        "pros":{
            "type":"string"
        },
        "cons":{
            "type":"string"
        },
        "rating":{
            "type":"integer"
        },
        "sentiment":{
            "type":"string"
        }
    },
    "required":["summary","pros","cons","rating","sentiment"]
}

model_struchured_output=model.with_structured_output(json_schema)
response=model_struchured_output.invoke("Give me a review of the movie Inception")
print(response)
