import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.main import create_app


def export_schema():
    app = create_app()
    openapi_schema = app.openapi()

    output_path = os.path.join(os.path.dirname(__file__), "..", "openapi.json")
    output_path = os.path.abspath(output_path)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)

    print(f"Successfully exported OpenAPI schema to {output_path}")


if __name__ == "__main__":
    export_schema()
