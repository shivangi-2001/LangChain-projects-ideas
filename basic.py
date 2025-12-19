from langchain.llms import OpenAI
import dotenv

dotenv.load_dotenv()

llm = OpenAI()

result = llm("Can you write code for video peer conferencing?")
print(result)

# notes:
# langchain -> openai -> [Http request {input prompt} -> openAI server]