from flask import Flask, request, jsonify, session
from agents.off_topic_agent import  create_off_topic_agent_app
from agents.medication_reminder_agent import create_medication_reminder_agent_app
from agents.agent_supervisor import query_router
from langchain_core.messages import HumanMessage, AIMessage
from agents.ethic_reviewer_agent import apply_ethics_after_response
import os
from flask_cors import CORS
import json
from datetime import datetime
from google.cloud import firestore
import random
import string
from flask_caching import Cache
from dotenv import load_dotenv

from agents.simulator_agent import generate_common_sense, get_llm_response
# from util import get_current_persona


app = Flask(__name__)
CORS(app)  # This will enable CORS for all routes
app.secret_key = 'ABC_TEST_1234'  # Required for session management

# Configure the cache
app.config['CACHE_TYPE'] = 'SimpleCache'  # In-memory cache
app.config['CACHE_DEFAULT_TIMEOUT'] = 360000  # 100 hours
cache = Cache(app)


# Authenticate to Firestore with the JSON account key.
db = firestore.Client.from_service_account_json("firestore-key.json")

def generate_response(input_text, agent, agent_type):
    simulator_action = ""
    final_state = agent.invoke(
    {"messages": [HumanMessage(content=input_text)]},
        config={"configurable": {"thread_id": thread_id}}
    )
    agent_response = final_state["messages"][-1].content

    # Store the response in session
    if "global_chat" not in session:
        session["global_chat"] = []
    session["global_chat"].append({
        "assistant": agent_response,
        "system": "Invoked Agent: " + agent_type
    })

    # return agent_response, simulator_action
    simulator_action = ""
    return agent_response, simulator_action


# Generate a random 5-character string
def generate_random_id(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))


@app.route('/api/conversation', methods=['POST'])
def perform_conversation():
    prompt = request.json.get('prompt')
    chat_source = request.json.get('chat_source', None)
    if(chat_source == "WEBCHAT"):
        prompt = prompt
    else:
        prompt = prompt[0]
    if prompt:
        # Determine which agent to use
        if "previous_agent" not in session:
            if chat_source == "WEBCHAT":
                selected_agent = query_router(prompt, chat_source="WEBCHAT")
            else:
                selected_agent = query_router(prompt)
        else:
            if chat_source == "WEBCHAT":
                selected_agent = query_router(prompt, previous_agent=session["previous_agent"], chat_source="WEBCHAT")
            else:
                selected_agent = query_router(prompt, previous_agent=session["previous_agent"])
        session["previous_agent"] = selected_agent

        if selected_agent == "OFF_TOPIC_AGENT":
            response = generate_response(prompt, off_topic_agent, selected_agent)
            response = response[0]
        elif selected_agent == "MEDICATION_REMINDER_AGENT":
            response = generate_response(prompt, medication_reminder_agent, selected_agent)
            response = response[0]
        elif selected_agent == "SIMULATOR_AGENT":
            current_profile_json = get_current_persona()
            is_user_visually_impaired = current_profile_json.get('is_visually_impaired')
            response = get_llm_response(prompt, is_user_visually_impaired)
        else:
            response = "No valid agent selected."
        simulator_action = ""
        return jsonify({"invoked_agent": selected_agent, "response": response, "simulator_action": simulator_action}), 200

    return jsonify({"error": "No prompt provided."}), 400

@app.route('/api/common_sense', methods=['POST'])
def get_common_sense():
    prompt = request.json.get('prompt')
    prompt = prompt[0]

    response = generate_common_sense(prompt)

    return jsonify({"response": response}), 200

@app.route('/api/add_persona', methods=['POST'])
def add_persona():
    """
    Add a new persona to the `persona` collection.

    Parameters:
        name (str): The name of the persona.
        medication (str): The medication associated with the persona.
        persona_description (str): A description of the persona.

    Returns:
        dict: The ID of the newly created persona document.
    """
    name = request.json.get('name')
    medication = request.json.get('medication')
    persona_description = request.json.get('persona_description')
    random_id = generate_random_id()
    is_visually_impaired = request.json.get('is_visually_impaired')

    persona_data = {
        "id": random_id,
        "name": name,
        "medication": medication,
        "persona_description": persona_description,
        "is_visually_impaired": is_visually_impaired
    }
    collection_ref = db.collection("persona")
    new_doc_ref = collection_ref.add(persona_data)
    return {"persona_id": new_doc_ref[1].id}

@app.route('/api/view_personas', methods=['GET'])
def view_personas():
    """
    Retrieve all personas from the `persona` collection.

    Returns:
        list: A list of persona documents.
    """
    collection_ref = db.collection("persona")
    docs = collection_ref.stream()
    personas = []
    for doc in docs:
        persona_data = doc.to_dict()
        persona_data["id"] = doc.id  # Add the document ID for reference
        personas.append(persona_data)
    return personas

@app.route('/api/make_persona_default', methods=['POST'])
def make_persona_default():
    """
    Update the default persona in the `basic_data` collection.

    Parameters:
        persona_id (str): The ID of the persona to set as default.

    Returns:
        dict: The updated basic_data document.
    """
    persona_id = request.json.get('persona_id')

    basic_data_ref = db.collection("basic_data").limit(1).get()
    if not basic_data_ref:
        raise ValueError("No basic_data record found.")

    for doc in basic_data_ref:
        doc_ref = doc.reference
        doc_ref.update({"default_persona": persona_id})
        current_profile_json = get_current_persona()
        cache.set('my_cached_data', current_profile_json)  # Update the cache

        return {"default_persona": persona_id}

@app.route('/api/get_current_persona', methods=['GET'])
def get_current_persona():
    """
    Retrieve the current default persona based on the `default_persona` value
    in the `basic_data` collection.

    Returns:
        dict: The persona data of the current default persona.
    """
    # Fetch the `basic_data` collection
    basic_data_ref = db.collection("basic_data").limit(1).get()
    if not basic_data_ref:
        raise ValueError("No basic_data record found.")

    for doc in basic_data_ref:
        basic_data = doc.to_dict()
        default_persona_id = basic_data.get("default_persona")

        if not default_persona_id:
            raise ValueError("default_persona not set in basic_data.")

        # Fetch the default persona document
        persona_ref = db.collection("persona").document(default_persona_id)
        persona_doc = persona_ref.get()

        if not persona_doc.exists:
            raise ValueError("Default persona not found in persona collection.")

        # Extract persona data
        persona_data = persona_doc.to_dict()
        persona_data["id"] = persona_doc.id  # Add the document ID for reference

        # Validate and parse JSON fields (e.g., 'medication') if present
        if "medication" in persona_data:
            medication_data = persona_data["medication"]
            try:
                if isinstance(medication_data, str):  # Parse JSON string
                    persona_data["medication"] = json.loads(medication_data)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in medication field: {e}")

        return persona_data

    raise ValueError("No persona found in the provided basic_data.")


@app.route('/api/edit_persona', methods=['POST'])
def edit_persona():
    """
    Edit an existing persona in the `persona` collection.

    Parameters:
        persona_id (str): The ID of the persona to edit.
        name (str, optional): The updated name of the persona.
        medication (str, optional): The updated medication of the persona.
        persona_description (str, optional): The updated description of the persona.

    Returns:
        dict: The updated persona document.
    """
    persona_id = request.json.get('id')
    name = request.json.get('name')
    medication = request.json.get('medication')
    persona_description = request.json.get('persona_description')
    is_visually_impaired = request.json.get('is_visually_impaired')
    
    persona_ref = db.collection("persona").document(persona_id)
    persona_doc = persona_ref.get()

    if not persona_doc.exists:
        raise ValueError("Persona with the given ID does not exist.")

    updates = {}
    if name is not None:
        updates["name"] = name
    if medication is not None:
        updates["medication"] = medication
    if persona_description is not None:
        updates["persona_description"] = persona_description
    if is_visually_impaired is not None:
        updates["is_visually_impaired"] = is_visually_impaired

    if updates:
        persona_ref.update(updates)

    updated_persona = persona_ref.get().to_dict()
    updated_persona["id"] = persona_id  # Add the document ID for reference
    return updated_persona

def setup_persona():
    global off_topic_agent
    global medication_reminder_agent
    global thread_id
    
    # Generate a random integer between 1 and 100 for thread id
    thread_id = random.randint(1, 100)

    current_profile_json = get_current_persona()
    medication_reminder_agent = create_medication_reminder_agent_app(cache)
    off_topic_agent = create_off_topic_agent_app(cache)
    

if __name__ == '__main__':
    setup_persona()
    app.run(debug=True)