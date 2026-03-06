import os
import re
import time
import sys
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

# 2. Define Tools (Mock Weather Function)
def get_weather(city):
    """Simulates an API call to get weather data."""
    # In a real scenario, this would request an external API
    if "tokyo" in city.lower():
        return '{"temp": 15, "condition": "rainy"}'
    elif "new york" in city.lower():
        return '{"temp": 20, "condition": "sunny"}'
    else:
        return '{"temp": 25, "condition": "cloudy"}'

# 3. Define System Prompt (The "Brain" instructions)
SYSTEM_PROMPT = """
You are a helpful AI assistant with access to a weather tool.
To answer the user's question, you must follow this Thought-Action-Observation loop:

1. **Thought**: Analyze what you need to do.
2. **Action**: If you need data, output exactly: `call: get_weather("CityName")`
3. **Observation**: **STOP! Do not generate this part.** I will give you the tool result.
4. ...Repeat until you have the answer...
5. **Final Answer**: You MUST start your final response with "Final Answer:" followed by the answer.

**Rules:**
- Do not make up or guess tool results.
- Wait for the user to provide the "Observation".
- If you have the answer, use "Final Answer:".

Available Tools:
- get_weather(city): Returns weather data (temp, condition).
"""

# 4. The Agent Loop (ReAct Logic)
def run_agent(user_query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print_log("USER", f"Query: {user_query}", Colors.BOLD)
    print("-" * 50)
    
    for step in range(5): # Max 5 steps to prevent infinite loops
        print_log("AGENT", f"Step {step + 1}: Thinking...", Colors.CYAN)
        
        # Call LLM
        start_time = time.time()
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=messages,
                temperature=0  # Deterministic for tools
            )
        except Exception as e:
            print_log("ERROR", f"API Call failed: {e}", Colors.YELLOW)
            return

        end_time = time.time()
        
        response_text = response.choices[0].message.content
        
        # Add agent response to history
        messages.append({"role": "assistant", "content": response_text})
        
        # Print the Agent's raw output (Thought & Action)
        print(f"{Colors.GREEN}{response_text}{Colors.RESET}\n")
        print_log("META", f"Time taken: {end_time - start_time:.2f}s", Colors.CYAN)
        
        # Check for "Final Answer"
        if "Final Answer" in response_text:
            return response_text
            
        # Check for "Action" (Regex to find `call: get_weather("...")`)
        # Supports both single and double quotes, and optional spaces
        match = re.search(r'call:\s*get_weather\s*\(\s*[\'"](.+?)[\'"]\s*\)', response_text)
        if match:
            city = match.group(1)
            print_log("TOOL", f"Executing: get_weather('{city}')", Colors.YELLOW)
            
            # Simulate Tool Execution
            observation = get_weather(city)
            print_log("OBSERVATION", f"Result: {observation}", Colors.YELLOW)
            print("-" * 50)
            
            # Feed observation back to LLM
            messages.append({
                "role": "user", 
                "content": f"Observation: {observation}"
            })
        else:
            # If no action and no final answer, maybe it's just thinking or asking for more info
            pass

if __name__ == "__main__":
    select_model()
    # Test the agent
    run_agent("What should I wear in New York today?")
