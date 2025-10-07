import hashlib

def make_id(text: str) -> str:
    """Create a deterministic ID from text using SHA1 hash."""
    return hashlib.sha1(text.strip().encode("utf-8")).hexdigest()
