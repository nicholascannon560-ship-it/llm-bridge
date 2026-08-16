"""Test syntax of modified llm-bridge files."""
import ast
import sys

files_to_check = [
    "llm_gateway.py",
    "agent_loop/tools.py",
    "command_channel.py",
]

errors = []
for fpath in files_to_check:
    try:
        with open(fpath, "r") as f:
            source = f.read()
        ast.parse(source, filename=fpath)
        print(f"✓ {fpath}: syntax OK")
    except SyntaxError as e:
        errors.append(f"✗ {fpath}: {e}")
        print(f"✗ {fpath}: {e}")
    except FileNotFoundError:
        errors.append(f"✗ {fpath}: not found")
        print(f"✗ {fpath}: not found (run from repo root)")

if errors:
    sys.exit(1)
print("\nAll files pass syntax check.")
