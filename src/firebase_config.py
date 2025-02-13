import firebase_admin
from firebase_admin import credentials, firestore
import os
import uuid  # For generating unique chat IDs
from datetime import datetime, timezone

# Check if Firebase is already initialized
if not firebase_admin._apps:
    cred = credentials.Certificate("firebase_key.json")
    firebase_admin.initialize_app(cred)

# Get Firestore client
db = firestore.client()


def create_user_if_not_exists(user_id, user_name, user_email):
    """Check if the user exists in Firestore. If not, create a new user document."""
    user_ref = db.collection("users").document(user_id)
    if not user_ref.get().exists:
        user_ref.set({
            "name": user_name,
            "email": user_email,
            "created_at": datetime.now(timezone.utc),
        })

def create_new_chat(user_id):
    """Creates a new chat session with placeholder data."""
    chat_id = str(uuid.uuid4())  # Generate a unique chat ID
    chat_ref = db.collection("users").document(user_id).collection("chats").document(chat_id)

    chat_ref.set({
        "created_at": datetime.utcnow(),
        "last_updated": datetime.utcnow(),
        "title": "New Chat",  # Placeholder title
        "messages": [
            {"role": "assistant", "content": "Hello, how can I help you today?"}
        ]
    })

    return chat_id  # Return the new chat ID