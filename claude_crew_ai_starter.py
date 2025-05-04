# Advanced Multi-Agent System using CrewAI
# pip install crewai langchain-openai langchain-community python-dotenv

import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun

# Load environment variables
load_dotenv()

# Initialize the LLM
llm = ChatOpenAI(
    model_name="gpt-4",
    temperature=0.7,
    api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize search tool
search_tool = DuckDuckGoSearchRun()

# Define the specialized agents
researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover comprehensive and accurate information on a given topic",
    backstory="""You are an expert research analyst with a knack for finding 
    relevant information. You're meticulous and always verify facts.""",
    verbose=True,
    llm=llm,
    tools=[search_tool]
)

writer = Agent(
    role="Content Writer",
    goal="Create engaging and informative content based on research",
    backstory="""You are a skilled writer who can turn complex information 
    into clear, engaging content. You excel at structuring information in a 
    reader-friendly way.""",
    verbose=True,
    llm=llm
)

reviewer = Agent(
    role="Content Reviewer",
    goal="Ensure content is accurate, comprehensive, and well-structured",
    backstory="""You have years of experience in editing and reviewing content. 
    You have an eye for detail and can spot inconsistencies and areas for 
    improvement quickly.""",
    verbose=True,
    llm=llm
)

# Define the tasks for each agent
def create_research_task(topic):
    return Task(
        description=f"""
        Research the topic of {topic} thoroughly. 
        Identify key aspects, recent developments, and important details.
        Compile your findings in a well-structured research report.
        """,
        agent=researcher,
        expected_output="A comprehensive research report on the topic"
    )

def create_writing_task():
    return Task(
        description="""
        Using the research provided, create an engaging article.
        Make sure to: 
        - Include an attention-grabbing introduction
        - Organize information logically with clear headings
        - Explain complex concepts in simple terms
        - End with a meaningful conclusion
        """,
        agent=writer,
        expected_output="A well-written article based on the research"
    )

def create_review_task():
    return Task(
        description="""
        Review the article thoroughly. Check for:
        - Factual accuracy
        - Clarity and flow
        - Grammar and spelling
        - Completeness of information
        
        Provide specific feedback and suggestions for improvement.
        """,
        agent=reviewer,
        expected_output="A detailed review with specific feedback"
    )

# Create the crew
def run_content_crew(topic):
    research_task = create_research_task(topic)
    writing_task = create_writing_task()
    review_task = create_review_task()
    
    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        verbose=2,
        process=Process.sequential  # Tasks will be executed in sequence
    )
    
    result = crew.kickoff()
    return result

# Example usage
if __name__ == "__main__":
    topic = input("Enter a topic for the agents to research and write about: ")
    print(f"\nCreating content about: {topic}")
    print("\nThis process might take a few minutes...\n")
    
    result = run_content_crew(topic)
    print("\n=== FINAL RESULT ===\n")
    print(result)