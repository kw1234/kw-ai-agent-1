# Simple AI Agent using LangChain
# pip install langchain langchain-openai langchain-community python-dotenv

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables (create a .env file with your API keys)
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4", 
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Define custom tools
@tool
def calculator(expression: str) -> float:
    """Evaluates a mathematical expression."""
    return eval(expression)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Define tools
tools = [
    calculator,
    search_tool,
]

# Create the agent prompt
prompt = PromptTemplate.from_template("""
You are an intelligent AI assistant. Your goal is to help users accomplish their tasks.
Use these tools to answer the user question:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original user question

Begin!

Question: {input}
Thought: 
""")

# Create the agent
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True,
    handle_parsing_errors=True
)

# Memory for the agent (for maintaining context)
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

# Example of agent usage
def run_agent(query):
    try:
        response = agent_executor.invoke({"input": query})
        return response["output"]
    except Exception as e:
        return f"Error: {str(e)}"

# Interactive mode
if __name__ == "__main__":
    print("AI Agent initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        response = run_agent(user_input)
        print(f"\nAgent: {response}")