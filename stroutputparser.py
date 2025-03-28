import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

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

prompt1 = template1.format(topic="black holes")  
result = model.invoke(prompt1) 
 

prompt2 = template2.format(text=result.content)  
result1 = model.invoke(prompt2)  


print(result1.content)
