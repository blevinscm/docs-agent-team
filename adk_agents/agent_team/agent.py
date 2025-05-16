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


async def generate_text(
    document_uri: str,
    document_mime_type: str,
   # file_part: types.Part = None, # Add this parameter for file input. Default to None.
    ):
    """
    Generates documentation text based on a given document URI,
    incorporating system instructions and a prompt template.

    Args:
        document_uri (str): The URI (e.g., a GitHub issue link) of the document to analyze.
        document_mime_type (str): The MIME type of the document (e.g., "text/html", "text/plain").


    Returns:
    dict: A dictionary with a 'status' key ('success' or 'error')
              and a 'report' key containing the generated documentation text in Markdown format
              (if successful), or an error message (if unsuccessful).
    """

    print(f"--- Tool: generate_text called : ---")
    #if file_part:
     #   print(f"--- Tool: received file: {document_uri} ---")

    client = genai.Client(
        vertexai=True,
        project="genai-docs-project",
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"

    si_text = f""" You are a technical writer specializing in Google Cloud documentation. 
              Your task is to generate documentation for Google Cloud services.  """


    prompt_template = f"""
        Look at the Github issue link, analyze the change request and adapt or create content that matches the document style and tone. 

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

        User Request: "{document_uri}"

        Your concise answer:
        """
    # --- Construct the 'contents' for the LLM call ---
    # This is the core part for handling multimodal input.
    # We will build the 'parts' list dynamically.
    parts_for_llm = [
        types.Part.from_text(text=prompt_template) # Start with the structured user prompt
    ]

    # Now create the final contents list
    contents = [
        types.Part.from_uri(
            file_uri=document_uri,
            mime_type=document_mime_type,
        ),
        types.Part.from_text(text="Make changes as asked in this issue document."),
    ]


    generate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        max_output_tokens = 8192,
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(text=si_text)
    ],
    )
    try:
        response = client.models.generate_content(
        model=model,
        config=generate_content_config,
        contents=contents,
        )
        generated_text = response.text
        return {"status": "success", "report": generated_text}

    except Exception as e:
        print(f"Error generating text with file input: {e}")
        return {"status": "error", "report": f"Failed to generate text from file: {e}"}


  
async def evaluate_text(
    document_uri: str,
    document_mime_type: str,
   # file_part: types.Part = None, # Add this parameter for file input. Default to None.
    ):
    """
    Evaluates documentation text based on a given document URI,
    incorporating system instructions and a prompt template for evaluation criteria.

    Args:
        document_uri (str): The URI (e.g., a GitHub issue link or a document URL)
                            of the document to be evaluated.
        document_mime_type (str): The MIME type of the document (e.g., "text/html", "text/plain").

    Returns:
        dict: A dictionary with a 'status' key ('success' or 'error')
              and a 'report' key containing the evaluation results (e.g., feedback,
              score, or identified issues) in a structured or Markdown format,
              or an error message if the evaluation was unsuccessful.
    """

    print(f"--- Tool: evaluate_text called : ---")
    #if file_part:
     #   print(f"--- Tool: received file: {document_uri} ---")

    client = genai.Client(
        vertexai=True,
        project="genai-docs-project",
        location="us-central1",
    )
    model = "gemini-2.0-flash-001"

    si_text = f""" You are an expert documentation evaluator for Google Cloud. 
              Your task is to thoroughly review and assess the quality, accuracy, clarity, and adherence to style guidelines of the provided documentation. 
              Identify areas for improvement and provide constructive feedback. """


    prompt_template = f"""
      **Documentation Evaluation Request:**

        Your task is to act as a highly critical and thorough documentation evaluator for Google Cloud. You will be provided with documentation content. Analyze this content meticulously based on the following criteria:

        1.  **Technical Accuracy & Groundedness:**
            * Is the information factually correct and up-to-date?
            * Are all technical concepts explained accurately?
            * Are code samples, commands, and configurations correct and executable?
            * Does it align with current product features and behavior?

        2.  **Clarity & Conciseness:**
            * Is the language clear, unambiguous, and easy to understand for the target audience (developers, data scientists, ML engineers)?
            * Are sentences and paragraphs concise?
            * Is jargon avoided or clearly explained?

        3.  **Completeness & Comprehensiveness:**
            * Does the document cover the topic adequately?
            * Are there any obvious information gaps or missing steps?
            * Are prerequisites, limitations, and potential edge cases addressed?

        4.  **Adherence to Style Guide & Tone:**
            * Does the content follow the Google Cloud style guide for voice, tone (e.g., clear, concise, professional, active voice)?
            * Is terminology used consistently and correctly (e.g., product names, feature names)?
            * Are formatting guidelines (headings, lists, code blocks) correctly applied?
            * Are there any typos, grammatical errors, or awkward phrasing?

        5.  **User Experience & Navigability (if applicable to content):**
            * Is the content logically structured and easy to navigate?
            * Are cross-references and internal links accurate and helpful?

        **Instructions:**

        Provide your evaluation in a clear, structured report. For each criterion, briefly state if it passes or fails and provide specific, actionable feedback or identified issues, referencing specific parts of the provided document where possible. If the document passes all criteria, state that it is approved.
        User Request: "{document_uri}"

        Your concise answer:
        """
        
    # --- Construct the 'contents' for the LLM call ---
    # This is the core part for handling multimodal input.
    # We will build the 'parts' list dynamically.
    parts_for_llm = [
        types.Part.from_text(text=prompt_template) # Start with the structured user prompt
    ]

    # Now create the final contents list
    contents = [
        types.Part.from_uri(
            file_uri=document_uri,
            mime_type=document_mime_type,
        ),
        types.Part.from_text(text="Make changes as asked in this issue document."),
    ]


    evaluate_content_config = types.GenerateContentConfig(
        temperature = 1,
        top_p = 1,
        max_output_tokens = 8192,
        safety_settings = [
            types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="OFF"),
            types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="OFF")
        ],
        system_instruction=[types.Part.from_text(text=si_text)
    ],
    )
    try:
        response = client.models.generate_content(
        model=model,
        config=evaluate_content_config,
        contents=contents,
        )
        evaluated_text = response.text
        return {"status": "success", "report": evaluated_text}

    except Exception as e:
        print(f"Error generating text with file input: {e}")
        return {"status": "error", "report": f"Failed to evaluate text from file: {e}"}   



# --- Greeting Agent ---
greeting_agent = Agent(
   # Using a potentially different/cheaper model for a simple task
   model = "gemini-2.0-flash",
   name="greeting_agent",
   instruction="You are the Greeting Agent. Your ONLY task is to provide a friendly greeting to the user. "
               "Use the 'say_hello' tool to generate the greeting. "
               "If the user provides their name, make sure to pass it to the tool. "
               "Do not engage in any other conversation or tasks.",
   description="Handles simple greetings and hellos using the 'say_hello' tool.", # Crucial for delegation
   tools=[say_hello],
)


# --- Farewell Agent ---
farewell_agent = Agent(
   # Can use the same or a different model
   model = "gemini-2.0-flash",
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
   name="generation_agent",
   instruction="You are the documentation generation  Agent. Your task is to provide documentation for a specified issue. "
               "Use the 'generate_text' tool when the user indicates they want documentation or docs for a specified issue. "
               "Do not perform any other actions.",
   description="Generates documents using the 'generate_text' tool.", # Crucial for delegation
   tools=[generate_text, load_artifacts],
)

 # --- Docs Evaluator Agent ---
evaluation_agent = Agent(
   # Can use the same or a different model
   model = "gemini-2.0-flash",
   name="evaluation_agent",
   instruction="You are the documentation evaluation Agent. Your task is to evaluate documentation for a specified issue. "
               "Use the 'evaluate_text' tool when the user indicates they want  documentation or docs to be evaluated for a specified issue. "
               "Do not perform any other actions.",
   description="Evaluates documents using the 'evaluate_text' tool.", # Crucial for delegation
   tools=[evaluate_text, load_artifacts],
)


   # @title Define the Root Agent with Sub-Agents
root_agent = Agent(
   name="infoport_supervisor_agent_v2", # Give it a new version name
   model="gemini-2.0-flash",
   description="The main Supervisor Agent. Its primary responsibility is to understand user intent and delegate requests to the appropriate specialized sub-agents.",
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
   tools=[], # A pure supervisor agent delegates all tasks and thus doesn't need its own tools
  
   sub_agents=[greeting_agent, farewell_agent, generation_agent, evaluation_agent]
)


