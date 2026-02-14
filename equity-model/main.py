"""
Entry point for the equity model.

Usage:
    python main.py init          Initialise (or re-initialise) the database
    python main.py reset         Drop and recreate all tables
"""

import sys

import yaml

from db.schema import init_db, reset_schema


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    db_path = config.get("database", {}).get("path", "data/equity.duckdb")

    if len(sys.argv) < 2:
        print("Usage: python main.py [init|reset]")
        sys.exit(1)

    command = sys.argv[1]

    if command == "init":
        con = init_db(db_path)
        tables = con.execute("SHOW TABLES").fetchall()
        print("Database initialised. Tables:", [t[0] for t in tables])
        con.close()

    elif command == "reset":
        con = reset_schema(db_path)
        tables = con.execute("SHOW TABLES").fetchall()
        print("Schema reset. Tables:", [t[0] for t in tables])
        con.close()

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
