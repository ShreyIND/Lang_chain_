import os
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader,TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

model = ChatGoogleGenerativeAI(

    model="models/gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
path=r"C:\Users\shrey\OneDrive\Documents\Desktop\web"
# The most stable way to load PDFs from a directory
loader = DirectoryLoader(
    path,
    glob="**/*.pdf",
    loader_cls=PyPDFLoader
)

docs = loader.load()
print(docs)
