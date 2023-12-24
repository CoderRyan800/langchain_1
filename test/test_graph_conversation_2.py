from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Neo4jVector

with open('neo4j_database.txt','r') as fp:
    raw_string = fp.readline()
    key_string = raw_string.strip()

loader = TextLoader("state_of_the_union.txt")

documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

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

db = Neo4jVector.from_documents(
    docs, OpenAIEmbeddings(), url=url, username=username, password=password
)

query = "What did the president say about Ketanji Brown Jackson"
docs_with_score = db.similarity_search_with_score(query, k=2)

for doc, score in docs_with_score:
    print("-" * 80)
    print("Score: ", score)
    print(doc.page_content)
    print("-" * 80)





