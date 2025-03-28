import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    model_kwargs={"api_key": os.getenv("HUGGINGFACE_API_KEY")} # Added api_key to model_kwargs
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

template = PromptTemplate(
    template="Give me the name, age and city of a fictional person \n {format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# By prompt
""""
prompt = template.format()
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)
print(type(final_result))

"""


# By chain
chain = template | model | parser
result = chain.invoke({})
print(result)


