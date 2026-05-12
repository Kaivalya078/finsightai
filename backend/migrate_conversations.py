"""
One-Time Migration: Assign Orphaned Conversations to a User
============================================================
Finds all conversations in MongoDB that have no user_id (created before
authentication was added) and assigns them to the specified user.

Usage:
    python migrate_conversations.py

The script will:
  1. List all users in the database
  2. Let you pick the target user
  3. Show how many orphaned conversations exist
  4. Ask for confirmation before updating
"""

import os
import sys
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "finsightai")


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    users_col = db["users"]
    convs_col = db["conversations"]

    # --- Step 1: List all users ---
    users = list(users_col.find({}, {"_id": 1, "name": 1, "email": 1}))
    if not users:
        print("❌ No users found in the database. Log in with Google OAuth first.")
        sys.exit(1)

    print("\n📋 Users in database:\n")
    for i, u in enumerate(users, 1):
        print(f"  {i}. {u.get('name', 'N/A')} ({u.get('email', 'N/A')})  →  _id: {u['_id']}")

    # --- Step 2: Pick target user ---
    if len(users) == 1:
        choice = 1
        print(f"\n→ Only one user found, auto-selecting: {users[0].get('email')}")
    else:
        try:
            choice = int(input(f"\nSelect user (1-{len(users)}): "))
        except (ValueError, EOFError):
            print("❌ Invalid input.")
            sys.exit(1)

    if choice < 1 or choice > len(users):
        print("❌ Invalid selection.")
        sys.exit(1)

    target_user = users[choice - 1]
    target_id = str(target_user["_id"])
    print(f"\n✅ Target user: {target_user.get('name')} ({target_user.get('email')})")
    print(f"   user_id: {target_id}")

    # --- Step 3: Count orphaned conversations ---
    orphan_filter = {
        "$or": [
            {"user_id": {"$exists": False}},
            {"user_id": None},
            {"user_id": ""},
        ]
    }
    orphan_count = convs_col.count_documents(orphan_filter)
    total_count = convs_col.count_documents({})

    print(f"\n📊 Conversations: {total_count} total, {orphan_count} orphaned (no user_id)")

    if orphan_count == 0:
        print("✅ No orphaned conversations found. Nothing to migrate.")
        sys.exit(0)

    # Show a preview
    print("\n📝 Orphaned conversations preview:")
    for doc in convs_col.find(orphan_filter, {"_id": 1, "title": 1, "created_at": 1}).limit(10):
        title = doc.get("title", "Untitled")[:60]
        created = doc.get("created_at", "unknown")
        print(f"   • {title}  (created: {created})")
    if orphan_count > 10:
        print(f"   ... and {orphan_count - 10} more")

    # --- Step 4: Confirm and migrate ---
    confirm = input(f"\n⚠️  Assign {orphan_count} conversations to {target_user.get('email')}? (y/N): ")
    if confirm.strip().lower() != "y":
        print("❌ Aborted.")
        sys.exit(0)

    result = convs_col.update_many(orphan_filter, {"$set": {"user_id": target_id}})
    print(f"\n✅ Migration complete! {result.modified_count} conversations updated.")
    print(f"   user_id set to: {target_id}")


if __name__ == "__main__":
    main()
