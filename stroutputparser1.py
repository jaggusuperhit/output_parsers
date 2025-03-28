import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


model = ChatOpenAI(
    model="mistralai/mistral-7b-instruct",  
    openai_api_key = os.getenv("OPENROUTER_API_KEY"), 
    openai_api_base="https://openrouter.ai/api/v1"
)

# 1st Prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", 
    input_variables=["topic"]
)

# 2nd Prompt -> Summary
template2 = PromptTemplate(
    template="Write a 5-line summary of the following text. /n{text}",
    input_variables=["text"]
)

parser = StrOutputParser()

chain = template1 | model | parser | template2 | model | parser

result = chain.invoke({"topic":"black holes"})
print(result)