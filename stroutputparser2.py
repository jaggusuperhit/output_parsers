import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    model_kwargs={"api_key": os.getenv("HUGGINGFACE_API_KEY")} # Added api_key to model_kwargs
)

model = ChatHuggingFace(llm=llm)


# 1st Prompt -> Detailed Report
template1 = PromptTemplate(
    template="Write a detailed report on {topic}", 
    input_variables=["topic"]
)

# 2nd Prompt -> Summary
template2 = PromptTemplate(
    template="Write a 5-line summary of the following text. \n{text}",
    input_variables=["text"]
)

prompt1 = template1.format(topic="black holes") 
result = model.invoke(prompt1) 

prompt2 = template2.format(text=result.content) 
result1 = model.invoke(prompt2) 

print(result1.content)