from langchain.llms import OpenAI
import dotenv

dotenv.load_dotenv()

llm = OpenAI()

result = llm("Write very very short poem.")
print(result)

# notes:
# langchain -> openai -> [Http request {input prompt} -> openAI server]