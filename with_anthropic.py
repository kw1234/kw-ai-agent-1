import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

api_key = os.environ["ANTHROPIC_API_KEY"]
model = "claude-3-5-sonnet-20240620"

# Initialize the LLM using Anthropic's Claude
llm = ChatAnthropic(model=model, api_key=api_key)

# Test the LLM
response = llm.invoke("Hello, world!")
print(response.content)

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

# Create the agent prompt - FIXED with agent_scratchpad
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
{agent_scratchpad}
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

# Example usage
def run_agent():
    print("AI Agent initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        try:
            response = agent_executor.invoke({"input": user_input})
            print(f"\nAgent: {response['output']}")
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again with a different query.")

# Run the agent if this file is executed directly
if __name__ == "__main__":
    run_agent()