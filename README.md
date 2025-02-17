# CoRDial Core

CoRDial offers a flexible and modular system for integrating new agents as tasks evolve. The Dialogue Manager oversees interactions, routing queries to specialized agents using the ProtoNet Router

## Prerequisites

Ensure you have the following installed before proceeding:
- **Python** (3.7+ recommended) – [Download here](https://www.python.org/downloads/)
- **pip** (comes with Python) or **pipenv** for dependency management
- **Firestore Database** (through Google Cloud)

## Setup Instructions

### 1. Clone the Repository
```sh
git clone https://github.com/cordial-ai/cordial-core.git
cd cordial-core
```

### 2. Install Dependencies
Ensure you're in the root directory of your project, then install the required dependencies:
```sh
pip install -r requirements.txt
```

Alternatively, if you are using `pipenv`:
```sh
pipenv install
```

### 3. Set up Firestore Database
1. Go to [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or use an existing project.
3. Enable Firestore for your project in the **Firestore** section.
4. Create a new service account.
5. Download the service account key file in JSON format (usually named something like `firestore-key.json`).
6. Add the downloaded `firestore-key.json` to the root directory of the project.

Please refer this turotial [Get started with Cloud Firestore](https://firebase.google.com/docs/firestore/quickstart)

### 4. Set Up Environment Variables

### 5. Run the Flask API Server
Now, you can run the Flask development server:

```sh
python app.py
```
Or, if you're using Flask's CLI commands:
```sh
flask run
```

The application should now be running at:  
🔗 **http://localhost:5000**

Now you're all set! 🎉
