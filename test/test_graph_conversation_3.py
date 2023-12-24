from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Neo4jVector

with open('neo4j_database.txt','r') as fp:
    raw_string = fp.readline()
    key_string = raw_string.strip()

# loader = TextLoader("state_of_the_union.txt")

# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# docs = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()

# Neo4jVector requires the Neo4j database credentials

url = "bolt://localhost:7687"
username = "neo4j"
password = key_string

# You can also use environment variables instead of directly passing named parameters
# os.environ["NEO4J_URI"] = "bolt://localhost:7687"
# os.environ["NEO4J_USERNAME"] = "neo4j"
# os.environ["NEO4J_PASSWORD"] = "pleaseletmein"

# The Neo4jVector Module will connect to Neo4j and create a vector index if needed.

# Now we initialize from existing graph
existing_graph = Neo4jVector.from_existing_graph(
    embedding=OpenAIEmbeddings(),
    url=url,
    username=username,
    password=password,
    index_name="it_index",
    node_label="SubSection",
    text_node_properties=["summary", "title"],
    embedding_node_property="embedding",
)
result = existing_graph.similarity_search("Regulations on keeping animals in the units.", k=3)

print(result)

# existing_graph = Neo4jVector.from_existing_graph(
#     embedding=OpenAIEmbeddings(),
#     url=url,
#     username=username,
#     password=password,
#     index_name="person_index",
#     node_label="Section",
#     text_node_properties=["summary", "title"],
#     embedding_node_property="embedding",
# )
# result = existing_graph.similarity_search("voting", k=1)

# print(result)

