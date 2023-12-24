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
    template = """You are creating a summary of input documents to be represented
    as entities and relationships in Neo4j Cypher.  You must extract the main high-level
    entities and their relationships from the new document and use them
    to re-write the existing graph so as
    to capture the most important entities and relationships in the existing graph
    and to introduce the main entities and relationships from the new document to
    the graph.  Output is a re-written graph that 
    must be written in pure Cypher because it will be read into
    Neo4j using automated tools.  For this reason, every Cypher state must end in
    a semicolon.  Under NO circumstances should any item that is not
    a pure Cypher statement be left uncommented because the user will read the
    final output directly into Neo4j!  It is absolutely unacceptable if even one
    line of output that is not pure Cypher is left uncommented.
    However, you must include good English comments so that human readers and yourself
    can understand the semantic meaning of the graph and use such knowledge to
    guide future updates, but you MUST comment out these comments so they do NOT
    interfere with automated loading of the data - do NOT leave them uncommented
    under any possible circumstances!
    Do not repeat variables and do not repeat entities
    that already exist.  You are not augmenting but actually re-writing the graph
    so please focus only on major entities and core relationships.  If the graph
    is blank, then you must write a new graph based on the document you have.  Be
    sure that your comments explain the entities and relationships so that you can
    read them and use them to guide future updates.

    Existing graph:
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

    #llm_model = ChatOpenAI(model='gpt-3.5-turbo-16k')
    llm_model = ChatOpenAI(model='gpt-4')

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
                print("\n\n%s\n\n" % (current_output[0],))
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
            #previous_output_string_buffer = previous_output_string_buffer + "\n%s\n" % (current_output[0],)
            # Here, we are completely re-writing the graph rather than augmenting it.
            previous_output_string_buffer = current_output[0]
        # End loop over documents
    # End with block for cb
    return list_of_outputs

# End function create_graph_from_documents

def scrub_graph_output(input_filename, output_filename):

    llm = OpenAI(temperature=0)
    # Notice that "chat_history" is present in the prompt template
    template = """
    Read in the existing text under existing graph.  

    1. Any statement that is not
    a syntactically valid Cypher sstatement must be commented out, parituclarly if
    it is a natural language statement.

    2. Check for and do your best to fix any syntax error in the Cypher statements.

    3. Fix any issues that would cause the input not to work properly when
    ingested into Neo4j.

    Existing graph:
    {existing_graph}

    Response:"""
    prompt = PromptTemplate.from_template(template)

    #llm_model = ChatOpenAI(model='gpt-3.5-turbo-16k')
    llm_model = ChatOpenAI(model='gpt-4')

    working_chain = prompt | llm_model | StrOutputParser()

    with open(input_filename,'r') as fp:
        document_string = fp.read()
    
    input_dictionary = {"existing_graph": document_string}

    output_list = working_chain.batch([input_dictionary])

    new_output = output_list[0]

    with open(output_filename,'w') as fp:
        fp.write(new_output)

# End function scrub_graph_output
