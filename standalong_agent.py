# Standalone AI Agent
# pip install openai requests beautifulsoup4

import os
import json
import requests
from bs4 import BeautifulSoup
import re
import datetime
from typing import List, Dict, Any, Optional

class Tool:
    """Base class for agent tools"""
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    def execute(self, input_text: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

class WebSearchTool(Tool):
    """Tool for web searching"""
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the web for information. Input should be a search query."
        )
    
    def execute(self, query: str) -> str:
        try:
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(f"https://www.google.com/search?q={query}", headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                results = []
                
                # Extract search results (simplified)
                for result in soup.select('div.tF2Cxc'):
                    title_elem = result.select_one('h3')
                    link_elem = result.select_one('a')
                    snippet_elem = result.select_one('div.IsZvec')
                    
                    if title_elem and link_elem and snippet_elem:
                        title = title_elem.get_text()
                        link = link_elem.get('href')
                        if link.startswith('/url?q='):
                            link = link.split('/url?q=')[1].split('&')[0]
                        snippet = snippet_elem.get_text()
                        results.append(f"Title: {title}\nURL: {link}\nSnippet: {snippet}\n")
                
                return "\n".join(results[:3])  # Return top 3 results
            else:
                return f"Error: Unable to complete search. Status code: {response.status_code}"
        except Exception as e:
            return f"Error during web search: {str(e)}"

class CalculatorTool(Tool):
    """Tool for performing calculations"""
    def __init__(self):
        super().__init__(
            name="calculator",
            description="Perform mathematical calculations. Input should be a mathematical expression."
        )
    
    def execute(self, expression: str) -> str:
        try:
            # Sanitize input for security
            if any(char in expression for char in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ;'\"\\"):
                return "Error: Invalid characters in expression"
            
            result = eval(expression)
            return f"Result: {result}"
        except Exception as e:
            return f"Error calculating result: {str(e)}"

class DateTimeTool(Tool):
    """Tool for getting current date and time"""
    def __init__(self):
        super().__init__(
            name="datetime",
            description="Get current date and time information."
        )
    
    def execute(self, input_text: str) -> str:
        now = datetime.datetime.now()
        return f"Current date: {now.strftime('%Y-%m-%d')}\nCurrent time: {now.strftime('%H:%M:%S')}"

class StandaloneAgent:
    """A simple standalone AI agent with tools"""
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tools = {
            "web_search": WebSearchTool(),
            "calculator": CalculatorTool(),
            "datetime": DateTimeTool(),
        }
        self.conversation_history = []
    
    def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI API to get a response"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    *self.conversation_history,
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            
            return response.choices[0].message.content
        except Exception as e:
            return f"Error calling LLM API: {str(e)}"
    
    def _get_system_prompt(self) -> str:
        """Generate the system prompt with tool descriptions"""
        tool_descriptions = "\n".join([f"- {name}: {tool.description}" for name, tool in self.tools.items()])
        
        return f"""
        You are a helpful AI assistant with access to the following tools:
        
        {tool_descriptions}
        
        To use a tool, respond with JSON in the following format:
        {{
            "thoughts": "your reasoning about what to do",
            "tool": "tool_name",
            "tool_input": "input for the tool"
        }}
        
        If you don't need to use a tool, respond with:
        {{
            "thoughts": "your reasoning",
            "final_answer": "your response to the user"
        }}
        """
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse the JSON response from the LLM"""
        try:
            # Extract JSON from the response
            json_match = re.search(r'```json\n(.*?)\n```', response, re.DOTALL)
            if json_match:
                response = json_match.group(1)
            
            # Clean up any markdown formatting
            response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
            
            # Try to find a JSON object in the text
            json_pattern = r'\{.*\}'
            json_match = re.search(json_pattern, response, re.DOTALL)
            if json_match:
                response = json_match.group(0)
            
            return json.loads(response)
        except Exception as e:
            # If we can't parse JSON, treat the entire response as a final answer
            return {
                "thoughts": "Failed to parse JSON",
                "final_answer": response
            }
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        """Execute the specified tool with the given input"""
        if tool_name in self.tools:
            return self.tools[tool_name].execute(tool_input)
        else:
            return f"Error: Tool '{tool_name}' not found"
    
    def process_query(self, user_input: str, max_steps: int = 5) -> str:
        """Process a user query, potentially using tools"""
        self.conversation_history.append({"role": "user", "content": user_input})
        
        for step in range(max_steps):
            # Get response from LLM
            llm_response = self._call_llm(user_input)
            parsed_response = self._parse_response(llm_response)
            
            # If we have a final answer, return it
            if "final_answer" in parsed_response:
                self.conversation_history.append({"role": "assistant", "content": parsed_response["final_answer"]})
                return parsed_response["final_answer"]
            
            # Otherwise, execute the tool
            if "tool" in parsed_response and "tool_input" in parsed_response:
                tool_name = parsed_response["tool"]
                tool_input = parsed_response["tool_input"]
                
                tool_result = self._execute_tool(tool_name, tool_input)
                
                # Add this step to the conversation
                self.conversation_history.append({
                    "role": "assistant", 
                    "content": f"I need to use the {tool_name} tool.\nThoughts: {parsed_response.get('thoughts', 'No thoughts provided')}"
                })
                
                # Create a new user message with the tool result
                user_input = f"Tool result: {tool_result}\nPlease continue."
                self.conversation_history.append({"role": "user", "content": user_input})
            else:
                # If the response doesn't contain tool info or final answer, treat it as a final answer
                self.conversation_history.append({"role": "assistant", "content": llm_response})
                return llm_response
        
        # If we've reached the maximum number of steps, return a fallback response
        fallback_response = "I've thought about this but haven't reached a conclusive answer yet. Let me provide my best response based on what I know."
        self.conversation_history.append({"role": "assistant", "content": fallback_response})
        return fallback_response

# Usage example
if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    agent = StandaloneAgent(api_key)
    
    print("AI Agent initialized. Type 'exit' to quit.")
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'exit':
            break
        
        response = agent.process_query(user_input)
        print(f"\nAgent: {response}")