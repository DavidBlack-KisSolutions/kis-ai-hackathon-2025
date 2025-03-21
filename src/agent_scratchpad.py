import json
import os
from typing import Dict

from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_google_vertexai import ChatVertexAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Set up Google credentials
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./credentials.json"

# Initialize the model
model = ChatVertexAI(model="gemini-1.5-pro-002")

# Database configuration
DATABASE_FILE = "./database.json"

INITIAL_DATABASE_FIELDS = [
    "relationshipStatus",
    "partnersName",
    "partnersFavoriteFood",
    "relationshipStartDate",
    "partnerBirthDate",
    "activitiesWithPartner",
]

# Initialize the database
database: Dict[str, str] = {}


def load_database() -> Dict[str, str]:
    """Load the database from JSON file or create a new one if it doesn't exist."""
    try:
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, "r") as f:
                return json.load(f)
        else:
            initial_database = {key: "" for key in INITIAL_DATABASE_FIELDS}
            with open(DATABASE_FILE, "w") as f:
                json.dump(initial_database, f, indent=2)
            return initial_database
    except json.JSONDecodeError:
        print("Could not load database")
        return {}


def save_database():
    """Save the current database state to JSON file."""
    with open(DATABASE_FILE, "w") as f:
        json.dump(database, f, indent=2)


# Database tools
@tool(description="Add a value to the database using a key")
def add_to_database(key: str, value: str):
    print(f"Adding {key} to database")
    database[key] = value
    save_database()


@tool(description="Update an existing value in the database using a key")
def update_database(key: str, value: str):
    print(f"Updating {key}: {value}")
    database[key] = value
    save_database()


@tool(description="Retrieve a value from the database using a key")
def get_from_database(key: str) -> str:
    print(f"Getting {key} from database")
    return database.get(key, "Key not found")


@tool(description="Get all the keys from the database")
def get_all_keys_from_database() -> list[str]:
    print("Getting all keys from the database")
    return list(database.keys())


@tool(description="Delete a value from the database using a key")
def delete_from_database(key: str):
    print(f"Deleting {key} from database")
    if key in database:
        del database[key]
        save_database()


@tool(description="Clear all the keys and values from the database")
def clear_database():
    print("Clearing all keys and values from the database")
    database.clear()
    save_database()


def create_database_agent():
    """Create and configure the database agent."""
    tools = [
        add_to_database,
        get_from_database,
        get_all_keys_from_database,
        delete_from_database,
        update_database,
    ]

    model_with_tools = model.bind_tools(tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""You are a helpful relationship assistant named Amy with access to a database.
    You can:
    - Add key-value pairs to the database
    - Retrieve values by key
    - List all keys in the database
    - Delete specific key-value pairs
    - Clear the entire database

    You cant:
    - Tell the user the key names for the database.

    Always use the tools to interact with the database, don't rely on your internal memory.

    Always infer the best key to use when creating new data, use camel case. First, list the keys to see if the key is available before inferring.

    Your First goal is to ensure the following keys have valid values retrieved from the user {INITIAL_DATABASE_FIELDS}
    Keep prompting the user for these values until they are all filled. If the answers are already filled, this goal is complete. If the goal is already complete, do not tell the user you have collected all the information you need

    Your second goal, is to help the user with their relationship problems. This can be done through constructive conversation, or giving advice to the user to help their relationship
        Examples of good advice
        - Preparing the user's partner's favorite meals
        - Planning dates and anniversaries for the user's partner
        - Remembering the user's partners birthday and planning dates
        - Suggest some recipies to cook for partner

    Now you need to store more information about the user`s relationship as key-value pairs.

    Always be polite and helpful when interacting with the database.""",
            ),
            ("placeholder", "{messages}"),
        ]
    )

    memory = MemorySaver()
    agent_executor = create_react_agent(
        model_with_tools,
        tools,
        prompt=prompt,
        checkpointer=memory,
    )

    return agent_executor


def invoke(agent_executor, utterance: str) -> str:
    """Invoke the database agent with a user utterance."""
    print(f"> {utterance}")
    thread_id = "db_agent_123"
    response = agent_executor.invoke(
        {"messages": [HumanMessage(content=utterance)]},
        config={"configurable": {"thread_id": thread_id}, "run_mode": "parallel"},
    )
    messages = response["messages"]
    ai_message = messages[-1].content
    return ai_message


def chat():
    """Interactive chat loop for the database agent."""
    agent_executor = create_database_agent()
    utterance = input(invoke(agent_executor, "Start the conversation")).strip()

    while True:
        if "q" == utterance.casefold():
            break
        response = invoke(agent_executor, utterance)
        print("bot: " + response)
        utterance = input(response).strip()

    print("Goodbye")


if __name__ == "__main__":
    # Initialize the database
    database = load_database()

    # Start the chat loop
    chat()
