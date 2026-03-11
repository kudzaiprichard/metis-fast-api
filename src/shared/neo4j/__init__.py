"""
Neo4j graph database access for FastAPI.

Lifecycle:
    - lifespan startup  → call connect(uri, username, password)
    - request handling   → Depends(get_neo4j_db) injects the instance
    - lifespan shutdown  → call close()
"""

from src.shared.neo4j.neo4j_graph_database import Neo4jGraphDatabase

_instance: Neo4jGraphDatabase | None = None


def connect(uri: str, username: str, password: str) -> Neo4jGraphDatabase:
    """Create, connect, and store the singleton. Called once at startup."""
    global _instance
    _instance = Neo4jGraphDatabase(uri, username, password)

    if not _instance.connect():
        _instance = None
        raise RuntimeError(
            f"Failed to connect to Neo4j at {uri}. "
            "Ensure Neo4j is running and credentials are correct."
        )

    return _instance


def close() -> None:
    """Close the connection. Called once at shutdown."""
    global _instance
    if _instance:
        _instance.close()
        _instance = None


def get_neo4j_db() -> Neo4jGraphDatabase:
    """FastAPI dependency — returns the live Neo4j instance."""
    if _instance is None:
        raise RuntimeError("Neo4j is not initialised. Check application startup.")
    return _instance


__all__ = ["connect", "close", "get_neo4j_db", "Neo4jGraphDatabase"]