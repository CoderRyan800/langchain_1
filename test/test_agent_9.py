import sys 
import json
import pickle
import traceback

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore import Wikipedia
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import TextLoader, PythonLoader

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

from langchain.agents.agent_toolkits import create_retriever_tool
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.document_loaders.generic import GenericLoader
from langchain.document_loaders.parsers import LanguageParser
from langchain.text_splitter import Language
from langchain.vectorstores import Chroma

from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper

from langchain.utilities import StackExchangeAPIWrapper

MEMORY_BINARY_PICKLE_FILENAME = "memory_state.pkl"

# Utility function for getting exception data to present to
# the LLM that runs the agent.

def get_exception_data(e, depth = 1):

    try:
        tb = e.__traceback__
        exception_data = traceback.format_exception(type(e),e,tb)
        exception_string_1 = json.dumps(exception_data,indent=4)
        exception_string_2 = "".join(traceback.TracebackException.from_exception(e).format())
        return_dict = {
            "exception_stack_trace": exception_string_2,
            "exception_data": exception_string_1
        }
        return(return_dict)
    except Exception as e2:
        if depth >= 1:
            get_exception_data(e2, depth=depth-1)
        else:
            traceback.print_exc()
            print("\nWARNING: MAX EXCEPTION DEPTH LIMIT EXCEEDED!")

# End method get_exception_data
            
def present_exception_data_to_agent(e, agent_executor_object):

    exception_dictionary = get_exception_data(e)
    exception_string = json.dumps(exception_dictionary,indent=4)
    message_to_agent = (
        "THIS IS AN AUTO GENERATED MESSAGE FROM THE EXCEPTION HANDLING CODE: \n" +
        "The following exception data have been generated during execution: \n" +
        "%s\n" +
        "Please read the information and examine your own source code using the tool provided for that purpose to advise on the problem and its possible solutions.\n" +
        "Please note the trace may yield the wrong line number so search the code in case the exception traceback data has the wrong line number(s) and point out possible problems." +
        "Do not ask the human to search the code for the error.  Scour the code for the error(s) yourself and point them out even if the stack trace lacks the correct line number!"
    ) % (exception_string,)

    agent_response = agent_executor_object.invoke({"input":message_to_agent})["output"]

    return_dict = {
        "exception_information": message_to_agent,
        "agent_response": agent_response
    }

    return return_dict

# End code to warn agent of exception in own code.

loader = PythonLoader("/Users/ryanmukai/Documents/github/langchain_1/test/test_agent_9.py")

documents = loader.load()
len(documents)
python_splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON, chunk_size=2000, chunk_overlap=200
)
texts = python_splitter.split_documents(documents)
len(texts)
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(texts, embeddings)

retriever = db.as_retriever()

docstore = DocstoreExplorer(Wikipedia())

tool_retriever = create_retriever_tool(
    retriever,
    "search_source_code",
    "Searches and returns your very source code defining yourself as an AI agent.",
)

google_search = GoogleSearchAPIWrapper()
google_search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=google_search.run,
)

google_search_tool.run("Obama's first name?")

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search",
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup",
    ),
    tool_retriever,
    google_search_tool
]

#memory = ConversationBufferMemory(memory_key="chat_history")



# Initialize the ConversationBufferMemory
try: 
    # Load the JSON file
    with open(MEMORY_BINARY_PICKLE_FILENAME, 'rb') as f:
        memory = pickle.load(f)
except:
    memory = ConversationBufferMemory(memory_key="chat_history")
    print("\nFile %s does not appear to be valid - starting new memory!\n" % (MEMORY_BINARY_PICKLE_FILENAME,))


llm = OpenAI(temperature=0, model_name="gpt-4-0613")
agent_executor = initialize_agent(
    tools,
    llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    #agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #agent=AgentType.REACT_DOCSTORE,
    verbose=True,
    memory=memory,
    handle_parsing_errors=True
)

#question = "Author David Chanoff has collaborated with a U.S. Navy admiral who served as the ambassador to the United Kingdom under which President?"
#react.run(question)

stop_flag = False

while not stop_flag:

    try:

        input_string = input("Please ask the agent a question: ")

        if input_string == 'STOP':
            stop_flag = True 
            continue 

        agent_executor.invoke({"input":input_string})["output"]

        # dummy_var = 1 / 0

    except Exception as e:
        exception_dictionary = present_exception_data_to_agent(e, agent_executor)
        print(json.dumps(exception_dictionary,indent=4))   

json_string = json.dumps(json.loads(memory.json()),indent=4)

fp = open('Agent_Memory.json','w')

fp.write(json_string)

fp.close()

# Pickle the memory as well!

fp = open(MEMORY_BINARY_PICKLE_FILENAME,'wb')

pickle.dump(memory,fp)

fp.close()
