from langchain.chains.llm import LLMChain
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()

chat = ChatOpenAI()

memory = ConversationBufferMemory(memory_key="messages", return_messages=True)

prompt = ChatPromptTemplate(
    input_varaibles = ['content', 'messages'],
    messages = [
        HumanMessagePromptTemplate.from_template("{content}"),
        MessagesPlaceholder(variable_name="messages")
    ]
)

chain = LLMChain(
    llm=chat,
    prompt=prompt,
    memory=memory,
)


while True:
    content = input('Enter content: ');

    result = chain({"content":content})

    print(result["text"])