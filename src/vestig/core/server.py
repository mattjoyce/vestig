"""Simple HTTP server for Vestig to avoid model reload overhead"""

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from vestig.core.cli import build_runtime
from vestig.core.config import load_config
from vestig.core.retrieval import search_memories


class VestigServer:
    """Vestig server that keeps embedding model in memory"""

    def __init__(self, config_path: str, host: str = "127.0.0.1", port: int = 8765):
        """Initialize server with config.

        Args:
            config_path: Path to vestig config file
            host: Server host (default: 127.0.0.1)
            port: Server port (default: 8765)
        """
        self.config_path = config_path
        self.host = host
        self.port = port

        print(f"Loading config from {config_path}...")
        self.config = load_config(config_path)

        print("Initializing runtime (this will load the embedding model)...")
        self.storage, self.embedding_engine, self.event_storage, self.tracerank_config = (
            build_runtime(self.config)
        )
        print(f"✓ Model loaded: {self.config['embedding']['model']}")
        print(f"✓ Database: {self.config['storage']['db_path']}")

    def handle_query(self, query: str, limit: int = 5, show_timing: bool = False) -> dict[str, Any]:
        """Handle a search/recall query.

        Args:
            query: Search query text
            limit: Number of results
            show_timing: Show timing breakdown

        Returns:
            Dictionary with results and metadata
        """
        results = search_memories(
            query=query,
            storage=self.storage,
            embedding_engine=self.embedding_engine,
            limit=limit,
            event_storage=self.event_storage,
            tracerank_config=self.tracerank_config,
            show_timing=show_timing,
        )

        return {
            "query": query,
            "limit": limit,
            "count": len(results),
            "results": [
                {
                    "id": memory.id,
                    "score": float(score),
                    "content": memory.content,
                    "created_at": memory.created_at,
                    "metadata": memory.metadata,
                }
                for memory, score in results
            ],
        }

    def start(self):
        """Start the HTTP server."""

        server_instance = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/query":
                    # Read request body
                    content_length = int(self.headers["Content-Length"])
                    body = self.rfile.read(content_length)
                    request_data = json.loads(body.decode("utf-8"))

                    # Extract parameters
                    query = request_data.get("query")
                    limit = request_data.get("limit", 5)
                    show_timing = request_data.get("show_timing", False)

                    if not query:
                        self.send_error(400, "Missing 'query' parameter")
                        return

                    try:
                        # Process query
                        result = server_instance.handle_query(query, limit, show_timing)

                        # Send response
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(json.dumps(result).encode("utf-8"))

                    except Exception as e:
                        self.send_error(500, f"Query error: {e}")

                elif self.path == "/health":
                    # Health check endpoint
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    response = {
                        "status": "healthy",
                        "model": server_instance.config["embedding"]["model"],
                        "db": server_instance.config["storage"]["db_path"],
                    }
                    self.wfile.write(json.dumps(response).encode("utf-8"))

                else:
                    self.send_error(404, "Endpoint not found")

            def log_message(self, format, *args):
                # Suppress default logging, we'll do our own
                pass

        server = HTTPServer((self.host, self.port), RequestHandler)
        print(f"\n{'='*60}")
        print(f"Vestig server running on http://{self.host}:{self.port}")
        print(f"{'='*60}")
        print("Endpoints:")
        print(f"  POST /query    - Search/recall memories")
        print(f"  POST /health   - Health check")
        print(f"\nPress Ctrl+C to stop")
        print(f"{'='*60}\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down server...")
            server.shutdown()
            self.storage.close()
            print("Server stopped.")
