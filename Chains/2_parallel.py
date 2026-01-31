from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel,Field
from typing import Optional,Literal
import os
from dotenv import load_dotenv
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers import StrOutputParser
model = ChatGoogleGenerativeAI(
    model="models/gemini-flash-latest", 
    google_api_key="AIzaSyCt6G-nMMomegApWWzZyv8ovTb8PAB3gC4"
)

parser=StrOutputParser()

text="""Generative AI is a fully managed Oracle Cloud Infrastructure service that provides a set of state-of-the-art, customizable large language models (LLMs) that cover a wide range of use cases, including chat, text generation, summarization, rerank, and creating text embeddings.

Use the playground, the API, or the CLI to try out the ready-to-use pretrained models or create and host your own fine-tuned custom models based on your own data on dedicated AI clusters.

The OCI Generative AI service includes the following foundational models for chat, rerank, and text embeddings.

Chat
Ask questions and get conversational responses through an AI chatbot.
Rerank
Input a query and a list of texts and get an ordered array with each text assigned a relevance score. The relevance score is how the model ranks the documents, that's, how well each text matches the query.
Embedding
Convert text to vector embeddings to use in applications for semantic searches, recommender systems, text classification, or text clustering.
Using the Pretrained Foundational Models
To get started, use the playground to try the pretrained foundational models. Run your prompts, adjust the parameters, update your prompts, and rerun the models until you're happy with the results. Then get the code from the Console and copy the code into your applications.


User workflow options for using the Generative AI service ready-to-use pretrained models.
Fine-Tuning the Pretrained Models
You can create a copy of a pretrained foundational model, add your own training dataset, and let the OCI Generative AI service fine-tune the model for you. OCI Generative AI uses dedicated AI clusters specially sized for fine-tuning. These clusters belong only to your tenancy. After your model is fine-tuned, you create an endpoint for the custom model and host that model on a dedicated AI cluster that's designed for hosting. When you create the hosting cluster, select the correct pretrained model from which the fine-tuned model is derived from.


User workflow options for using the Generative AI service for fine-tuning a pretrained model and hosting the fine-tuned model through an endpoint.
Use Cases
Use the OCI Generative AI service for the following types of use cases.

Text Generation
Use the pretrained chat models or text generation models to create text for any purpose, for example:

Pitch for a new product
Slogan for a marketing campaign
Sales email to a client
Social media post
Job description
Title for an article
Conversation
You can ask questions in natural language and optionally submit text such as documents, emails, and product reviews to the LLM and the LLM reasons over the text and provides intelligent answers.

Data Extraction
Extract specific pieces of data from text, for example:

Extract applicant information from an application written in free-form text.
Extract dates or sums from a contract.
Extract insights or trends from data tables.
Summarization
Generate executive summaries for documents that are too long to read, or summarize any type of text, for example:

Documents
Contracts
Emails
Articles
Blog posts
Product reviews
Social media posts
Classification
Classify text into predefined categories, for example:

Given a list of support tickets, classify them by the department that should handle them.
Given a list of sectors and company names, classify the companies by their respective sectors.
Style Transfer
Change the style or tone of text, for example:

Rewrite any text in a different style, format (list or paragraphs), or tone.
Rephrase text.
Suggest grammatical improvements.
Semantic Similarity
Evaluate several inputs based on how similar their meaning is, for example:

Evaluate a list of questions sent to a support system to extract the most relevant answer given to similar questions in the past when a new question comes in.
Replace keyword-based searches with semantic searches to improve search results relevance.
Regions with Generative AI
Oracle hosts its OCI services in regions and availability domains. A region is a localized geographic area, and an availability domain is one or more data centers in that region.

 Important

For a complete list of available regions, see Generative AI Regions and to find out which models are available in a region near you.Generative AI Models by Region t
Services that Call into the Generative AI Service
 Important

See Generative AI Regions.
Accessing Generative AI in the Console
Sign in to the Console by using a supported browser.
In the navigation bar of the Console, select a region with Generative AI, for example, US Midwest (Chicago). See which models are offered in your region.
Open the navigation menu  and select Analytics & AI. Under AI Services, select Generative AI."""
parser=StrOutputParser()
prompt1=PromptTemplate(
    template="Give the 5 important points of the {text}",
    input_variables=["text"]
)

prompt2=PromptTemplate(
    template="Give the joke on {text} with the heading JOKE",
    input_variables=["text"]
)
chain=RunnableParallel(
    {
        "important_points":prompt1 | model | parser,
        "faq":prompt2 | model | parser
    }
)
prompt3=PromptTemplate(
    template="Merge the {important_points} and {faq} to give a summary of the ",
    input_variables=["important_points","faq"]
)

chain2=prompt3 | model | parser
end_chain=chain|chain2

response=end_chain.invoke({"text":text})
print(response)