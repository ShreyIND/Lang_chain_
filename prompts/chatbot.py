from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
import os
load_dotenv()  

model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

chat_history = [
    SystemMessage(content='You are a helpful AI assistant.')
]
while True:
    user_input = input('You: ')
    if user_input.lower() in ['exit', 'quit']:
        break
    chat_history.append(HumanMessage(content=user_input))
    
    try:
        result = model.invoke(chat_history)
        chat_history.append(AIMessage(content=result.content))
        print(f"AI: {result.content}\n")
    except Exception as e:
        print(f"An error occurred: {e}")
        chat_history.pop()

print(chat_history)
