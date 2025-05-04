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

os.environ["OPENAI_API_KEY"] = "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"
llm = ChatOpenAI()
llm.invoke("Hello, world!")