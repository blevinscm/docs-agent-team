import datetime
from zoneinfo import ZoneInfo
from google.adk.agents import Agent


# @title Import necessary libraries
import os
import asyncio
from google.adk.agents import Agent
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types
from google import genai
import base64 # For creating message Content/Parts
from google.genai import Client
from google.adk.tools import ToolContext
from google.adk.tools import load_artifacts




import warnings
# Ignore all warnings
warnings.filterwarnings("ignore")

#client = Client()

# @title Define the get_weather Tool
def get_weather(city: str) -> dict:
   """Retrieves the current weather report for a specified city.


   Args:
       city (str): The name of the city (e.g., "New York", "London", "Tokyo").


   Returns:
       dict: A dictionary containing the weather information.
             Includes a 'status' key ('success' or 'error').
             If 'success', includes a 'report' key with weather details.
             If 'error', includes an 'error_message' key.
   """
   print(f"--- Tool: get_weather called for city: {city} ---") # Log tool execution
   city_normalized = city.lower().replace(" ", "") # Basic normalization


   # Mock weather data
   mock_weather_db = {
       "newyork": {"status": "success", "report": "The weather in New York is sunny with a temperature of 25°C."},
       "london": {"status": "success", "report": "It's cloudy in London with a temperature of 15°C."},
       "tokyo": {"status": "success", "report": "Tokyo is experiencing light rain and a temperature of 18°C."},
   }


   if city_normalized in mock_weather_db:
       return mock_weather_db[city_normalized]
   else:
       return {"status": "error", "error_message": f"Sorry, I don't have weather information for '{city}'."}


# @title Define Tools for Greeting and Farewell Agents


# Ensure 'get_weather' from Step 1 is available if running this step independently.
# def get_weather(city: str) -> dict: ... (from Step 1)


def say_hello(name: str = "there") -> str:
   """Provides a simple greeting, optionally addressing the user by name.


   Args:
       name (str, optional): The name of the person to greet. Defaults to "there".


   Returns:
       str: A friendly greeting message.
   """
   print(f"--- Tool: say_hello called with name: {name} ---")
   return f"Hello, {name}!"


def say_goodbye() -> str:
   """Provides a simple farewell message to conclude the conversation."""
   print(f"--- Tool: say_goodbye called ---")
   return "Goodbye! Have a great day."

async def generate_text(user_prompt: str, tool_context: 'ToolContext'):
    """Generates text based on the user_prompt."""

    print(f"--- Tool: generate_text called ---")
    client = genai.Client(
        vertexai=True,
        project="genai-docs-project",
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"
    
    si_text = f""" You are a technical writer specializing in Google Cloud documentation. 
              Your task is to generate documentation for Google Cloud services, adhering strictly to the Google Cloud style guide. """

    prompt_template = f"""
        Generate documentation for Google Cloud services, ensuring it aligns with the Google Cloud style guide available at: https://cloud.google.com/vertex-ai/generative-ai/docs/overview.

        Follow these guidelines:

        1.  **Refer to the Style Guide:**
            *   Thoroughly review the Google Cloud style guide to understand the required tone, formatting, and content guidelines.
            *   Pay close attention to sections on voice and tone, terminology, code samples, and formatting.
        2.  **Adapt the Tone:**
            *   Maintain a clear, concise, and professional tone.
            *   Use active voice and avoid jargon unless necessary.
        3.  **Follow Formatting Guidelines:**
            *   Use appropriate headings, subheadings, and bullet points to organize content.
            *   Ensure code samples are properly formatted and syntax-highlighted.
        4.  **Use Terminology Correctly:**
            *   Adhere to the terminology and naming conventions specified in the style guide.
            *   Use consistent terminology throughout the documentation.
        5.  **Provide Clear Examples:**
            *   Include practical examples and use cases to illustrate concepts.
            *   Ensure examples are accurate and up-to-date.
        6.  **Review and Revise:**
            *   Carefully review the generated documentation for accuracy, clarity, and adherence to the style guide.
            *   Revise as needed to meet the required standards.
                Please analyze the following request and provide a clear, direct answer:

        User Request: "{user_prompt}"

        Your concise answer:
        """
    contents = [
        types.Content(
        role="user",
        parts=[
            types.Part.from_text(text=prompt_template)
        ]
        ),
    ]


    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        max_output_tokens = 8192,
        safety_settings = [types.SafetySetting(
        category="HARM_CATEGORY_HATE_SPEECH",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
        threshold="OFF"
        ),types.SafetySetting(
        category="HARM_CATEGORY_HARASSMENT",
        threshold="OFF"
        )],
         system_instruction=[types.Part.from_text(text=si_text)
    ],
    )
    response = client.models.generate_content(
    model=model,
    contents=contents,
    )
    return (response.text)




# --- Greeting Agent ---
greeting_agent = Agent(
   # Using a potentially different/cheaper model for a simple task
   model = "gemini-2.0-flash",
   # model=LiteLlm(model=MODEL_GPT_4O), # If you would like to experiment with other models
   name="greeting_agent",
   instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
               "Use the 'say_hello' tool to generate the greeting. "
               "If the user provides their name, make sure to pass it to the tool. "
               "Do not engage in any other conversation or tasks.",
   description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
   tools=[say_hello],
)
#print(f"✅ Agent '{greeting_agent.name}' created using model '{greeting_agent.model}'.")



# --- Farewell Agent ---
farewell_agent = Agent(
   # Can use the same or a different model
   model = "gemini-2.0-flash",
   # model=LiteLlm(model=MODEL_GPT_4O), # If you would like to experiment with other models
   name="farewell_agent",
   instruction="You are the Farewell Agent. Your ONLY task is to provide a polite goodbye message. "
               "Use the 'say_goodbye' tool when the user indicates they are leaving or ending the conversation "
               "(e.g., using words like 'bye', 'goodbye', 'thanks bye', 'see you'). "
               "Do not perform any other actions.",
   description="Handles simple farewells and goodbyes using the 'say_goodbye' tool.", # Crucial for delegation
   tools=[say_goodbye],
)
 
 # --- Docs Generator Agent ---
generation_agent = Agent(
   # Can use the same or a different model
   model = "gemini-2.0-flash",
   # model=LiteLlm(model=MODEL_GPT_4O), # If you would like to experiment with other models
   name="generation_agent",
   instruction="You are the documentation generation  Agent. Your task is to provide documentation for a specified issue. "
               "Use the 'generate_text' tool when the user indicates they want documentation or docs for a specified issue. "
               "Do not perform any other actions.",
   description="Generates documents using the 'generate_text' tool.", # Crucial for delegation
   tools=[generate_text, load_artifacts],
)

   # @title Define the Root Agent with Sub-Agents

root_agent = Agent(
   name="infoport_agent_v2", # Give it a new version name
   model="gemini-2.0-flash",
   description="The main coordinator agent. Handles github issues requests and delegates greetings/farewells/generation and evlauation to specialists.",
   instruction="You are the main Supervisor Agent coordinating a team. Your primary responsibility is to provide weather information. "
               "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
               "You have specialized sub-agents: "
               "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
               "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
               "3. 'generation_agent': Handles generating documentation content for Cloud AI docs. Delegate to it for these. "
               "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
               "If it is a documentation request to create or generate docs, delegate to 'generation_agent'. "
               "If it's a weather request, handle it yourself using 'get_weather'. "
               "For anything else, respond appropriately or state you cannot handle it.",
   tools=[get_weather], # Root agent still needs the weather tool for its core task
   # Key change: Link the sub-agents here!
   sub_agents=[greeting_agent, farewell_agent, generation_agent]
)


