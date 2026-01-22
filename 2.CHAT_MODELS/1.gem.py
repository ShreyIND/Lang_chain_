import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = ChatGoogleGenerativeAI(

    model="models/gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
try:
    response = llm.invoke("What is the capital of America?")
    print(response.content)
except Exception as e:
    print(f"Error: {e}")
