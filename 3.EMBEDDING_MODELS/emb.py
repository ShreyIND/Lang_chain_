import os
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

texts = [
    "What is the meaning of life?",
    "What is the purpose of existence?",
    "How do I bake a cake?",
]

vectors = embeddings_model.embed_documents(texts)

df = pd.DataFrame(
    cosine_similarity(vectors),
    index=texts,
    columns=texts,
)

print(df)
# from google import genai
# from google.genai import types
# import pandas as pd
# from sklearn.metrics.pairwise import cosine_similarity
# from dotenv import load_dotenv
# import os
# load_dotenv()
# client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# texts = [
#     "What is the meaning of life?",
#     "What is the purpose of existence?",
#     "How do I bake a cake?",
# ]

# result = client.models.embed_content(
#     model="gemini-embedding-001",
#     contents=texts,
#     config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
# )
# df = pd.DataFrame(
#     cosine_similarity([e.values for e in result.embeddings]),
#     index=texts,
#     columns=texts,
# )

# print(df)