from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory


llm = OpenAI(temperature=0)
# Notice that "chat_history" is present in the prompt template
template = """Please read in all relevant entities from the previous conversation and from
the new document string and return updates to the Neo4j Cypher graph.  Write outputs
in pure Cypher and describe relationships between key entities from the previous conversation
and from the new document.

Previous conversation:
{chat_history}

New document: {new_document}
Response:"""
prompt = PromptTemplate.from_template(template)
# Notice that we need to align the `memory_key`
memory = ConversationBufferMemory(memory_key="chat_history")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory
)

list_of_docs = [
    {
        "new_document": "Igor is dating Natasha and must buy her flowers from the florist."
    },
    {
        "new_document": "Natasha likes roses and orchids but dislikes carnations."
    },
    {
        "new_document": "The florist has orchids and carnations but not roses."
    },
    {
        "new_document": "Igor has a sister named Svetlana."
    },
    {
        "new_document": "Igor was born in Dnipro, Ukraine."
    },
    {
        "new_document": "Natasha was born in Lviv, Ukraine."
    }
]

list_of_outputs = []

for current_document in list_of_docs:
    current_output = conversation(current_document)
    print(current_output['text'])
    list_of_outputs.append(current_output)


