from langchain.chains.llm import LLMChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain.prompts import HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

chat = ChatOpenAI()

# Correct way to store memory in a file
# memory = ConversationBufferMemory(
#     chat_memory=FileChatMessageHistory("message.json"),
#     memory_key="message",
#     return_messages=True
# )

# Memory store in summary that auto summaries past interaction
memory = ConversationSummaryMemory(
    llm=chat,
    memory_key="message",
    return_messages=True
)

# Prompt template with message history and new user input
prompt = ChatPromptTemplate(
    input_variables=["content", "message"],
    messages=[
        MessagesPlaceholder(variable_name="message"),
        HumanMessagePromptTemplate.from_template("{content}")
    ]
)

chain = LLMChain(
    llm = chat,
    prompt = prompt,
    output_key='result',
    memory=memory,
    verbose=True
)

while True:
    content = input(">> ")
    result = chain.invoke({ "content": content })
    print(result["result"])