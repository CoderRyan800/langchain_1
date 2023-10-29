import json
import traceback
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter

def create_graph_from_documents(list_of_documents):

    llm = OpenAI(temperature=0)
    # Notice that "chat_history" is present in the prompt template
    template = """Please read in named entities and main ideas from the previous conversation and from
    the new document string and return updates to the Neo4j Cypher graph.  
    Add new entities and relationships that you find in the document to the graph.
    Write outputs
    in pure Cypher. Do not declare any variables already declared in other Cypher statements.
    Do not create an
    entity again that is already created.  Include the main relationships among named
    entities and key concepts in the graph. CRITICAL: Please add a semicolon after each Cypher
    statement because this will be read into neo4j using the :source command.

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

    llm_model = ChatOpenAI(model='gpt-3.5-turbo-16k')
    #llm_model = ChatOpenAI(model='gpt-4')

    working_chain = prompt | llm_model | StrOutputParser()

    list_of_outputs = []

    previous_output_string_buffer = ""

    document_index = 0

    with get_openai_callback() as cb:

        for current_document in list_of_documents:
            if type(current_document) is not dict:
                input_dictionary = {
                    "new_document": current_document.page_content,
                    "chat_history": previous_output_string_buffer
                }
                doc_string = current_document.page_content
            else:
                input_dictionary = {
                    "new_document": current_document["new_document"],
                    "chat_history": previous_output_string_buffer
                }
                doc_string = current_document["new_document"]
            try:
                current_output = working_chain.batch([input_dictionary])
                print("Document number %d\n" % (document_index,))
                print(cb)

                
            except Exception as e:
                print("Exception has occured when adding a new document.\n")
                print("DOCUMENT:\n%s\n" % (doc_string,))
                print("Document index = %d\n\n" % (document_index,))
                traceback.print_exc()
                print(cb)

                return list_of_outputs

            document_index = document_index + 1

            #print(current_output['text'])
            list_of_outputs.append(current_output[0])
            previous_output_string_buffer = previous_output_string_buffer + "\n%s\n" % (current_output[0],)
        # End loop over documents
    # End with block for cb
    return list_of_outputs

# End function create_graph_from_documents
