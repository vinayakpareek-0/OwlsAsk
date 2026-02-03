import pymongo
# from pymongo.server_api import ServerApi
from datetime import datetime
import uuid
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

class AuditLogger:
    def __init__(self, atlas_uri, db_name="support_bot"):
        """
        atlas_uri: Your mongodb+srv:// string from the Atlas dashboard
        """
        # Set the Stable API version for long-term compatibility
        self.client = pymongo.MongoClient(
            atlas_uri, 
            server_api=ServerApi('1')
        )
        self.db = self.client[db_name]
        self.collection = self.db["chat_logs"]
        
        # Test connection
        try:
            self.client.admin.command('ping')
            print("Successfully connected to MongoDB Atlas!")
        except Exception as e:
            print(f"Atlas Connection Failed: {e}")

    def log_interaction(self, session_id, user_id, user_query, bot_response, metadata=None):
        log_entry = {
            "session_id": session_id,
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "interaction": {
                "user": user_query,
                "bot": bot_response
            },
            "metadata": metadata or {}
        }
        self.collection.insert_one(log_entry)

    def get_user_history(self, user_id):
        """Finds all sessions for a specific customer."""
        return list(self.collection.find({"user_id": user_id}).sort("timestamp", -1))