import json
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

def create_graph_from_documents(list_of_documents):

    llm = OpenAI(temperature=0)
    # Notice that "chat_history" is present in the prompt template
    template = """Please read in all relevant entities from the previous conversation and from
    the new document string and return updates to the Neo4j Cypher graph.  Write outputs
    in pure Cypher and describe relationships between key entities from the previous conversation
    and from the new document.  The previous conversation consists of previous Cypher statements
    which you can use as context to create new statements to describe entity relationships based
    on both old statements and the new document.

    Previous conversation:
    {chat_history}

    New document: {new_document}
    Response:"""
    prompt = PromptTemplate.from_template(template)
    # Notice that we need to align the `memory_key`
    #memory = ConversationBufferMemory(memory_key="chat_history")
    conversation = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=True
    )

    llm_model = ChatOpenAI()

    working_chain = prompt | llm_model | StrOutputParser()

    list_of_outputs = []

    previous_output_string_buffer = ""

    for current_document in list_of_documents:
        if type(current_document) is not dict:
            input_dictionary = {
                "new_document": current_document.page_content,
                "chat_history": previous_output_string_buffer
            }
        else:
            input_dictionary = {
                "new_document": current_document["new_document"],
                "chat_history": previous_output_string_buffer
            }
        current_output = working_chain.batch([input_dictionary])
        #print(current_output['text'])
        list_of_outputs.append(current_output[0])
        previous_output_string_buffer = previous_output_string_buffer + "\n%s\n" % (current_output[0],)

    return list_of_outputs

# End function create_graph_from_documents
