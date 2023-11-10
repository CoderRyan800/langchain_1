import json
import traceback
from operator import itemgetter
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda

template = """You are a nice chatbot having a conversation with a human.

Previous conversation:
{chat_history}

New human question: {question}
Response:"""

prompt = PromptTemplate.from_template(template)

llm_model = ChatOpenAI(model='gpt-4')

memory = ConversationBufferMemory(memory_key="chat_history")

conversation = LLMChain(
    llm=llm_model,
    prompt=prompt,
    verbose=True,
    memory=memory
)

working_chain = (
    conversation 
    | StrOutputParser()
)

stop_flag = False

while not stop_flag:

    input_words = input("Please enter text: ")

    if input_words == 'STOP':
        stop_flag = True
        break

    input_dictionary = {
        "question": input_words
    }

    ai_output = conversation.invoke(input_dictionary)

    print("\nAI RESPONSE: %s\n" % (ai_output['text'],))

# End while loop

memory_json = memory.json()

fp = open('chat_memory_dump.json','w')

fp.write(json.dumps(memory_json,indent=4))

fp.close()
