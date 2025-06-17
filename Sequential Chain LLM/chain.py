# langchain provide tools to automate each steps of a text generation pipeline
# make it easy to connect tools together

# chain to make reusable text-generation pipelines
# chains can be connected together to male a more complex pipeline
# chains consist to components promptTemplate and an LLM

# prompt template = input text we sending to do the llm model with variable
# llm - chatgpt, gemini, bard

# after getting output from the first llm pass to second llm model for verification

from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import dotenv
import argparse

dotenv.load_dotenv()

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="list of numbers")
parser.add_argument("--language", default="C++")

args = parser.parse_args()

llm = OpenAI()

code_prompt = PromptTemplate(
    input_variables=['language', 'task'],
    template="Write a very short {language} function that will {task}",
)

# output without second chain [test chain]
# {'language': 'Python', 'task': 'list of fruits are found in japan', 'text': '\n\n\ndef fruits_in_japan():\n    return ["apple", "banana", "grape", "strawberry"]'}
code_chain = LLMChain(
    llm = llm,
    prompt = code_prompt,
    output_key='code'
)
# In the sequential chain the code_chain pass the code and language to the test_chain model
# that generate the testing code for teh previous code for code_chain

test_prompt = PromptTemplate(
    input_variables=["language", "code"],
    template="Write a test for the following {language} code:\n{code}"
)

test_chain = LLMChain(
    llm = llm,
    prompt = test_prompt,
    output_key='test'
)

chain = SequentialChain(
    chains=[code_chain, test_chain],
    input_variables=['language', 'task'],
    output_variables=['test', 'code']
)

result = chain({
    "language": args.language,
    "task": args.task
})

print(result["code"])
print()
print(result["test"])