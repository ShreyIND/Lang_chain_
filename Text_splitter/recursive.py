from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

path = "2512.24601v1.pdf"
loader = PyPDFLoader(path)
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", " ", ""]
)

split_docs = splitter.split_documents(docs)

prompt = PromptTemplate(
    template="Summarize the following text into 1 important bullet point:\n\n{text}",
    input_variables=["text"]
)

model = ChatGoogleGenerativeAI(
    model="gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chain = prompt | model | StrOutputParser()

print(f"Total chunks created: {len(split_docs)}\n")

for i, doc in enumerate(split_docs[:5]):
    print(f"--- Chunk {i+1} Summary ---")
    result = chain.invoke({"text": doc.page_content})
    print(result, "\n")