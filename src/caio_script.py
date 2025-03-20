import json
import os
from typing import Dict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import the Gemini wrapper
# from gemini_wrapper import default_gemini

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "../notebookForAiHackathon/credentials.json"

model = ChatVertexAI(model="gemini-1.5-pro-002")  # gemini-1.5-flash

# Database file path
DATABASE_FILE = "./database.json"


# Initialize database from JSON file if it exists, otherwise create empty dict
def load_database() -> Dict[str, str]:
    if os.path.exists(DATABASE_FILE):
        try:
            with open(DATABASE_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}


def save_database():
    with open(DATABASE_FILE, "w") as f:
        json.dump(database, f, indent=2)


# Initialize the database
database: Dict[str, str] = load_database()

@tool(description="Add a value to the database using a key")
def add_to_database(key: str, value: str):
    database[key] = value
    save_database()

@tool(description="retrieve a value from the database using a key")
def get_from_database(key: str) -> str:
    return database.get(key, "Key not found")

@tool(description="Get all the keys from the database")
def get_all_keys_from_database() -> list[str]:
    return list(database.keys())

@tool(description="Delete a value from the database using a key")
def delete_from_database(key: str):
    if key in database:
        del database[key]
        save_database()


def clear_database():
    database.clear()
    save_database()


def create_database_agent():

    # Create the list of tools
    tools = [add_to_database, get_from_database, get_all_keys_from_database, delete_from_database]

    # Initialize the model using the Gemini wrapper
    model_with_tools = model.bind_tools(tools)

    # Create the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful assistant named Amy with access to a database.
        You can:
        - Add key-value pairs to the database
        - Retrieve values by key
        - List all keys in the database
        - Delete specific key-value pairs
        - Clear the entire database

        Always use the tools to interact with the database, don't rely on your internal memory.
        
        Always be polite and helpful when interacting with the database.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    # Create memory and agent
    memory = MemorySaver()
    agent_executor = create_react_agent(
        model_with_tools, tools, prompt=prompt, checkpointer=memory
    )

    return agent_executor


def invoke(utterance: str) -> str:
    """Invoke the database agent with a user utterance."""
    thread_id = "db_agent_123"
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content=utterance)]},
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = response["messages"]
    ai_message = messages[-1].content
    return ai_message


def chat():
    """Interactive chat loop for the database agent."""
    while True:
        utterance = input(">").strip()
        if "q" == utterance.casefold():
            break
        print(invoke(utterance))
    print("Goodbye")


if __name__ == "__main__":
    # Create the agent
    agent_executor = create_database_agent()

    # Start the chat loop
    print(
        "Database Agent: Hello! I can help you manage the database. What would you like to do?"
    )
    chat()