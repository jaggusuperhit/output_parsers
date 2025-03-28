import os
from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    model_kwargs={"api_key": os.getenv("HUGGINGFACE_API_KEY")} # Added api_key to model_kwargs
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1',  description='Fact 1 about the topic'),
    ResponseSchema(name='fact2',  description='Fact 2 about the topic'),
    ResponseSchema(name='fact3',  description='Fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template="Give 3 facts about {topic} \n {format_instruction}",
    input_variables=["topic"],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)
# By prompt
"""
prompt = template.invoke({"topic" : "Italy"})
result = model.invoke(prompt)
final_result = parser.parse(result.content)
print(final_result)"

"""

# By chain
chain = template | model | parser
result = chain.invoke({"topic" : "Italy"})
print(result)
