from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from langchain.document_loaders import TextLoader
from langchain.chains import GraphQAChain

# Load the text document
with open('document.txt', 'r') as f:
    text = f.read()

# Create the graph from the text document
index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
graph = index_creator.from_text(text)

# Create the GraphQAChain with the graph and LLM
chain = GraphQAChain.from_llm(llm=OpenAI(temperature=0), graph=graph)

# Ask a question about the graph
question = 'What is the capital of France?'
answer = chain.run(question)
print(answer)

