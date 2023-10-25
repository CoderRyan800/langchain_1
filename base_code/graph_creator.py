import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

def create_graph_from_documents(list_of_documents):

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

    list_of_outputs = []

    for current_document in list_of_documents:
        current_output = conversation(current_document)
        #print(current_output['text'])
        list_of_outputs.append(current_output['text'])

    return list_of_outputs

# End function create_graph_from_documents
