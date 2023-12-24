import json

from operator import itemgetter

from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough

model = ChatOpenAI()
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful chatbot"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

chain = (
    RunnablePassthrough.assign(
        history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
    )
    | prompt
    | model
)

# inputs = {"input": "hi im bob"}
# response = chain.invoke(inputs)
# print(response)

# memory.save_context(inputs, {"output": response.content})

# memory.load_memory_variables({})

# inputs = {"input": "whats my name"}
# response = chain.invoke(inputs)
# print(response)

stop_flag = False 

while not stop_flag:
    input_string = input("Please enter text for the AI, or 'STOP' to exit: ")
    if input_string == 'STOP':
        stop_flag = True
        with open('chat_memory_dump.json','w') as fp:
            fp.write(json.dumps(memory.to_json(),indent=4))
        exit()
    else:
        inputs = {"input": input_string}
        response = chain.invoke(inputs)
        memory.save_context(inputs, {"output": response.content})
        print(response)