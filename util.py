import json
from google.cloud import firestore

db = firestore.Client.from_service_account_json("firestore-key.json")


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

