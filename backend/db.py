"""
FinSight AI - MongoDB Connection
=================================
Provides database connection and collection handles.

Collections:
  - users:         User accounts (name, email, password_hash)
  - conversations: Per-user chat history with messages

Author: FinSight AI Team
"""

import os
from pymongo import MongoClient, ASCENDING, DESCENDING
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "finsightai")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# ---------------------------------------------------------------------------
# Collections
# ---------------------------------------------------------------------------

users_collection = db["users"]
conversations_collection = db["conversations"]

# ---------------------------------------------------------------------------
# Indexes (idempotent — safe to call on every startup)
# ---------------------------------------------------------------------------

def ensure_indexes():
    """Create required indexes. Call once during app startup."""
    # Unique email for users
    users_collection.create_index("email", unique=True)

    # Fast user-scoped conversation lookups
    conversations_collection.create_index("user_id")
    conversations_collection.create_index(
        [("user_id", ASCENDING), ("updated_at", DESCENDING)]
    )

    print("📦 MongoDB indexes ensured")
