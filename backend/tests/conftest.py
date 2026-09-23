import os

# auth.py refuses to import without a real secret; give the test process one.
os.environ.setdefault("JWT_SECRET", "test-only-secret-" + "x" * 40)
# Never let tests reach a real database; MongoClient connects lazily anyway.
os.environ.setdefault("MONGO_URI", "mongodb://localhost:27017")
