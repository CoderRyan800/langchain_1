from langchain.chains import GraphCypherQAChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph

with open('neo4j_database.txt','r') as fp:
    raw_string = fp.readline()
    key_string = raw_string.strip()

graph = Neo4jGraph(
    url="bolt://localhost:7687", username="neo4j", password=key_string
)

memory = ConversationBufferMemory(return_messages=True)

chain = GraphCypherQAChain.from_llm(
    ChatOpenAI(temperature=0), graph=graph, verbose=True,
    memory=memory
)

test_result = chain.run('Do any nodes in the graph concern pets?')

print(test_result)
