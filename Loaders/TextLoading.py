import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(

    model="models/gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)



loader = TextLoader(r"Loaders\content.txt") 
docs = loader.load()

prompt = PromptTemplate(
    template="Give me 5 important points from this text: {text}",
    input_variables=["text"]
)

chain = prompt | model | StrOutputParser()

result = chain.invoke({"text": docs[0].page_content})
print(result)
