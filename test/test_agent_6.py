from langchain.agents import AgentType, Tool, initialize_agent
from langchain.agents.react.base import DocstoreExplorer
from langchain.docstore import Wikipedia
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory

docstore = DocstoreExplorer(Wikipedia())
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

question = "My name is Ryan.  Who was the first President of the United States of America?"
agent_executor.invoke({"input":question})["output"]

question = "As of today, 23 December 2023, with whom is Israel at war, and how did that war start and when?"
agent_executor.invoke({"input":question})["output"]

question = "What is my name?"
agent_executor.invoke({"input":question})["output"]


