import os
import json
import requests
import inspect
import logging
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# --- Configure Logging ---
# You can change this to logging.INFO to hide the raw JSON payloads,
# or logging.DEBUG to see everything.
# In a real project, this might be controlled by an env var: os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=logging.DEBUG, 
    format='%(message)s' # Keep it clean for the CLI
)
logger = logging.getLogger(__name__)

# --- 1. Define Native Python Functions (The Tools) ---
def get_weather(city: str) -> str:
    """Get the current weather for a specific city.
    
    Args:
        city: The name of the city, e.g. Tokyo, New York
    """
    logger.info(f"\n[TOOL EXECUTION] Getting weather for: {city}")
    if "tokyo" in city.lower():
        return json.dumps({"temp": 15, "condition": "rainy"})
    elif "new york" in city.lower():
        return json.dumps({"temp": 20, "condition": "sunny"})
    else:
        return json.dumps({"temp": "unknown", "condition": "unknown"})

def search_web(query: str) -> str:
    """Search the web for general knowledge, news, or facts using Brave Search.
    
    Args:
        query: The search query string
    """
    logger.info(f"\n[TOOL EXECUTION] Searching web for: {query}")
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key or api_key == "your_brave_api_key_here":
        logger.error("Error: BRAVE_API_KEY is not configured.")
        return "Error: BRAVE_API_KEY is not configured."

    try:
        url = "https://api.search.brave.com/res/v1/web/search"
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": api_key
        }
        params = {"q": query, "count": 3}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        results = response.json().get("web", {}).get("results", [])
        if not results:
            return "No relevant results found."
            
        snippets = [f"[{i+1}] {item.get('title')}: {item.get('description')}" for i, item in enumerate(results)]
        return "\n".join(snippets)
    except Exception as e:
        return f"Error occurred during web search: {str(e)}"

# --- 2. Initialize OpenAI Client (Since we are using Ark/DeepSeek) ---
# Note: Google ADK (google-genai) is hardcoded to expect Google's API endpoints 
# and authentication headers. It does not natively support OpenAI-compatible 
# endpoints (like Ark) without monkey-patching or using their specific Vertex AI paths.
# To demonstrate the framework value while keeping your Ark DeepSeek models, 
# we should use LangChain or OpenAI SDK's built-in abstractions. 
# But let's try a clever workaround first using the official OpenAI client 
# to mimic the ADK simplicity.

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
MODEL_ID = os.getenv("DEEPSEEK_MODEL")

# --- 3. Create the Agent using Swarm-like Abstraction ---
# Since Google ADK's HTTP client hardcodes authentication headers that break Ark,
# let's use a very popular lightweight Agent framework approach (similar to OpenAI Swarm).
# It gives the same "pass a function" experience as Google ADK.

def generate_schema(func):
    """
    A magic function that reads a Python function's signature and docstring,
    and automatically generates the JSON schema required by OpenAI.
    This is EXACTLY what LangChain/PydanticAI do under the hood!
    """
    sig = inspect.signature(func)
    
    # Very basic parsing for demonstration (assumes all params are strings)
    properties = {}
    required = []
    
    for param_name, param in sig.parameters.items():
        properties[param_name] = {"type": "string"}
        if param.default == inspect.Parameter.empty:
            required.append(param_name)
            
    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__.split('\n')[0] if func.__doc__ else "",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }
    }

def run_framework_agent(user_query: str):
    logger.info(f"\n{'='*50}\n[USER] Query: {user_query}\n{'='*50}")
    
    # In modern lightweight frameworks, you just map names to functions
    available_functions = {
        "get_weather": get_weather,
        "search_web": search_web
    }
    
    # We no longer hand-write JSON! We generate it dynamically.
    tools_schema = [generate_schema(func) for func in available_functions.values()]

    messages = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": user_query}
    ]
    
    # The "Framework" Loop Engine
    step = 1
    while True:
        logger.info(f"\n[AGENT] Step {step}: Thinking...")
        
        # Log exactly what is sent to LLM (Using DEBUG level)
        payload = {
            "model": MODEL_ID,
            "messages": messages,
            "tools": tools_schema
        }
        logger.debug("\n[DEBUG: REQUEST TO LLM] Payload being sent:")
        logger.debug(f"{json.dumps(payload, indent=2, default=str)}")
        
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            tools=tools_schema
        )
        
        # Log exactly what is received from LLM (Using DEBUG level)
        logger.debug("\n[DEBUG: RESPONSE FROM LLM] Raw Response Object:")
        logger.debug(f"{json.dumps(response.model_dump(), indent=2)}")
        
        msg = response.choices[0].message
        messages.append(msg)
        
        if not msg.tool_calls:
            logger.info(f"\n[FINAL ANSWER] {msg.content}")
            break
            
        for tool_call in msg.tool_calls:
            func_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            
            # The Framework Magic: Dynamically calling your function
            function_to_call = available_functions.get(func_name)
            if function_to_call:
                result = function_to_call(**args)
                logger.info(f"[OBSERVATION] Result: {result}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "name": func_name,
                    "content": str(result)
                })
        step += 1

if __name__ == "__main__":
    run_framework_agent("What is the weather of the capital of Japan?")
