import os
import re
from dotenv import load_dotenv
from openai import OpenAI

# 1. Load environment variables
load_dotenv()
client = OpenAI(
    api_key=os.getenv("DOUBAO_API_KEY"),
    base_url=os.getenv("DOUBAO_BASE_URL")
)
MODEL_ID = os.getenv("DOUBAO_MODEL")

# 2. Define Tools (Mock Weather Function)
def get_weather(city):
    """Simulates an API call to get weather data."""
    print(f"\n[Tool Call] get_weather(city='{city}')")
    # Hardcoded mock data for demonstration
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
3. **Observation**: I will give you the tool result.
4. ...Repeat until you have the answer...
5. **Final Answer**: Output your final response to the user.

Available Tools:
- get_weather(city): Returns weather data (temp, condition).
"""

# 4. The Agent Loop (ReAct Logic)
def run_agent(user_query):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    print(f"User: {user_query}")
    
    for _ in range(5): # Max 5 steps to prevent infinite loops
        # Call LLM
        response = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=0  # Deterministic for tools
        )
        response_text = response.choices[0].message.content
        print(f"\nAgent: {response_text}")
        
        # Add agent response to history
        messages.append({"role": "assistant", "content": response_text})
        
        # Check for "Final Answer"
        if "Final Answer" in response_text:
            return response_text
            
        # Check for "Action" (Regex to find `call: get_weather("...")`)
        # Supports both single and double quotes, and optional spaces
        match = re.search(r'call:\s*get_weather\s*\(\s*[\'"](.+?)[\'"]\s*\)', response_text)
        if match:
            city = match.group(1)
            observation = get_weather(city)
            print(f"Observation: {observation}")
            
            # Feed observation back to LLM
            messages.append({
                "role": "user", 
                "content": f"Observation: {observation}"
            })
        else:
            # If no action and no final answer, maybe it's just thinking or asking for more info
            # In a strict ReAct loop, we might prompt it to continue, but here we just loop.
            pass

if __name__ == "__main__":
    # Test the agent
    run_agent("What should I wear in Tokyo today?")
