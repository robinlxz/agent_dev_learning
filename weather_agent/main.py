import os
import json
import time
import sys
import requests
from dotenv import load_dotenv
from openai import OpenAI

# --- 0. Observability (Colors & Logs) ---
class Colors:
    CYAN = '\033[96m'   # Info / Thinking
    YELLOW = '\033[93m' # Tool / Action
    GREEN = '\033[92m'  # Success / Result
    BOLD = '\033[1m'
    RESET = '\033[0m'

def print_log(stage, message, color=Colors.CYAN):
    print(f"{color}[{stage}] {message}{Colors.RESET}")

# 1. Load environment variables
load_dotenv()

# Global variables (initialized in select_model)
client = None
MODEL_ID = None

def select_model():
    """Allows user to select which model to use."""
    global client, MODEL_ID
    
    print(f"{Colors.BOLD}Select AI Model:{Colors.RESET}")
    print("1. Doubao (Ark)")
    print("2. DeepSeek (Ark)")
    
    # Check for command line arguments first
    if len(sys.argv) > 1:
        choice = sys.argv[1]
        print(f"Enter choice (1 or 2): {choice}")
    else:
        try:
            choice = input("Enter choice (1 or 2): ").strip()
        except EOFError:
            print_log("ERROR", "Input not supported in this environment. Defaulting to Doubao.", Colors.YELLOW)
            choice = "1"
    
    if choice == "1":
        print_log("SYSTEM", "Selected: Doubao (Ark)", Colors.GREEN)
        api_key = os.getenv("DOUBAO_API_KEY")
        base_url = os.getenv("DOUBAO_BASE_URL")
        MODEL_ID = os.getenv("DOUBAO_MODEL")
    elif choice == "2":
        print_log("SYSTEM", "Selected: DeepSeek (Ark)", Colors.GREEN)
        api_key = os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        MODEL_ID = os.getenv("DEEPSEEK_MODEL")
    else:
        print_log("ERROR", "Invalid choice. Defaulting to Doubao.", Colors.YELLOW)
        api_key = os.getenv("DOUBAO_API_KEY")
        base_url = os.getenv("DOUBAO_BASE_URL")
        MODEL_ID = os.getenv("DOUBAO_MODEL")
        
    client = OpenAI(api_key=api_key, base_url=base_url)

# 2. Define Tools
def get_weather(city):
    """Simulates an API call to get weather data."""
    # In a real scenario, this would request an external API
    if "tokyo" in city.lower():
        return json.dumps({"temp": 15, "condition": "rainy"})
    elif "new york" in city.lower():
        return json.dumps({"temp": 20, "condition": "sunny"})
    else:
        return json.dumps({"temp": "unknown", "condition": "unknown"})

def search_web(query):
    """Real web search using Brave Search API."""
    print_log("TOOL_INTERNAL", f"Searching web for: {query}", Colors.YELLOW)
    
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key or api_key == "your_brave_api_key_here":
        # 不再 fallback，而是明确抛出异常或返回明确的错误信息让 LLM 知道工具不可用
        error_msg = "Error: BRAVE_API_KEY is not configured or is invalid. Real web search is unavailable."
        print_log("ERROR", error_msg, Colors.YELLOW)
        return error_msg

    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }
        params = {"q": query, "count": 3} # Limit to top 3 results to save token context
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract snippets from the top results
        results = data.get("web", {}).get("results", [])
        if not results:
            return "No relevant results found on the web."
            
        snippets = []
        for i, item in enumerate(results):
            title = item.get("title", "No Title")
            desc = item.get("description", "No Description")
            snippets.append(f"[{i+1}] {title}: {desc}")
            
        # Return a concatenated string of snippets for the LLM to read
        return "\n".join(snippets)
        
    except Exception as e:
        print_log("ERROR", f"Brave Search API failed: {e}", Colors.YELLOW)
        return f"Error occurred during web search: {str(e)}"

# Tool Definitions for OpenAI API
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a specific city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "The name of the city, e.g. Tokyo, New York"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search the web for general knowledge, news, or facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query string"
                    }
                },
                "required": ["query"]
            }
        }
    }
]

# 3. Define System Prompt (Simplified for Function Calling)
SYSTEM_PROMPT = """
You are a helpful AI assistant.
Use your general knowledge for simple facts, but use search_web for real-time news.
"""
# SYSTEM_PROMPT = """
# You are a helpful AI assistant.
# Use the available tools to answer the user's question. 
# Always verify facts using the `search_web` tool, even if you think you know the answer.
# """

# 4. The Agent Loop (Function Calling Logic)
def run_agent(user_query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print_log("USER", f"Query: {user_query}", Colors.BOLD)
    print("-" * 50)
    
    for step in range(3): # Reverted to 5 steps
        print_log("AGENT", f"Step {step + 1}: Thinking...", Colors.CYAN)
        
        # Log EXACTLY what is being sent to the LLM
        print_log("DEBUG: REQUEST TO LLM", "Payload being sent:", Colors.BOLD)
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0
        }
        # Print using json.dumps for clear formatting. 
        # Note: messages might contain objects (like previous responses) so we use a custom default handler.
        print(f"{Colors.YELLOW}{json.dumps(payload, indent=2, default=str)}{Colors.RESET}\n")

        # Call LLM with tools
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                tools=tools,
                tool_choice="auto", # Let model decide
                temperature=0
            )
        except Exception as e:
            print_log("ERROR", f"API Call failed: {e}", Colors.YELLOW)
            return

        end_time = time.time()
        
        # Log EXACTLY what was received from the LLM
        print_log("DEBUG: RESPONSE FROM LLM", "Raw Response Object:", Colors.BOLD)
        # response.model_dump() converts the pydantic object to a dictionary
        print(f"{Colors.GREEN}{json.dumps(response.model_dump(), indent=2)}{Colors.RESET}\n")
        
        response_message = response.choices[0].message
        
        # Add agent response to history (crucial for function calling flow)
        messages.append(response_message)
        
        # Check if model wants to call a tool
        tool_calls = response_message.tool_calls
        
        if tool_calls:
            # Model wants to call tools
            print(f"{Colors.GREEN}Thought: I need to call tools.{Colors.RESET}\n")
            print_log("META", f"Time taken: {end_time - start_time:.2f}s", Colors.CYAN)
            
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                print_log("TOOL", f"Executing: {function_name}({function_args})", Colors.YELLOW)
                
                # Execute tool
                tool_output = None
                if function_name == "get_weather":
                    tool_output = get_weather(function_args.get("city"))
                elif function_name == "search_web":
                    tool_output = search_web(function_args.get("query"))
                
                print_log("OBSERVATION", f"Result: {tool_output}", Colors.YELLOW)
                
                # Send tool result back to model
                messages.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "name": function_name,
                    "content": str(tool_output)
                })
            print("-" * 50)
        else:
            # Model has Final Answer (no tool calls)
            print(f"{Colors.GREEN}{response_message.content}{Colors.RESET}\n")
            print_log("META", f"Time taken: {end_time - start_time:.2f}s", Colors.CYAN)
            return response_message.content

if __name__ == "__main__":
    select_model()
    # Test cases
    run_agent("What is the weather of the capital of Japan?")
