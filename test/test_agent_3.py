# First, we import all the necessary modules and functions.
from langchain.chat_models import ChatOpenAI
from langchain.agents import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import AIMessage, HumanMessage
from langchain.tools.render import format_tool_to_openai_function
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain.agents import AgentExecutor

# We create an instance of the ChatOpenAI class with the model "gpt-3.5-turbo" and temperature 0.
# This will be our language model for generating responses.
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# We define a tool that calculates the length of a word.
# This tool is a function that takes a string as input and returns its length.
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)

# We create a list of tools. In this case, it only contains the get_word_length tool.
tools = [get_word_length]

# We define a key for storing the chat history in memory.
MEMORY_KEY = "chat_history"

# We create a template for the chat prompt.
# The template includes a system message, the chat history, a user message, and an agent scratchpad.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are very powerful assistant, but bad at calculating lengths of words.",
        ),
        MessagesPlaceholder(variable_name=MEMORY_KEY),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# We initialize the chat history as an empty list.
chat_history = []

# We bind the tools to the language model.
# This allows the language model to use the tools when generating responses.
llm_with_tools = llm.bind(functions=[format_tool_to_openai_function(t) for t in tools])

# We define the agent.
# The agent takes an input, a scratchpad, and a chat history, and generates a response using the prompt and the language model with tools.
agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_function_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    | llm_with_tools
    | OpenAIFunctionsAgentOutputParser()
)

# We create an executor for the agent.
# The executor will run the agent and manage the tools.
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# We define an input message.
input1 = "how many letters in the word educa?"

# We invoke the agent with the input message and the chat history.
# The agent generates a response and we store the result.
result = agent_executor.invoke({"input": input1, "chat_history": chat_history})

# We update the chat history with the input message and the agent's response.
chat_history.extend(
    [
        HumanMessage(content=input1),
        AIMessage(content=result["output"]),
    ]
)

# We invoke the agent again with a new input message and the updated chat history.
agent_executor.invoke({"input": "is that a real word?", "chat_history": chat_history})
