import json

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

memory = ConversationBufferMemory(memory_key="chat_history")

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

    input_string = input("Please ask the agent a question: ")

    if input_string == 'STOP':
        stop_flag = True 
        continue 

    agent_executor.invoke({"input":input_string})["output"]

json_string = json.dumps(json.loads(memory.json()),indent=4)

fp = open('Agent_Memory.json','w')

fp.write(json_string)

fp.close()
