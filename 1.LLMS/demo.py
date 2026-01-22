import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()
llm = GoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)
try:
    response = llm.invoke("What is the capital of America?")
    print(response) 
except Exception as e:
    print(f"Error: {e}")




