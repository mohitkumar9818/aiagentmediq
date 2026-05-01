"""Debug launcher — captures ALL output from app.py startup."""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

result = subprocess.run(
    [sys.executable, "app.py"],
    capture_output=True,
    text=True,
    timeout=12,  # Flask should bind within 12s
    env={**os.environ}
)

with open("debug_out.txt", "w") as f:
    f.write("=== STDOUT ===\n")
    f.write(result.stdout or "(empty)")
    f.write("\n=== STDERR ===\n")
    f.write(result.stderr or "(empty)")
    f.write(f"\n=== EXIT CODE: {result.returncode} ===\n")

print("STDOUT:", result.stdout[:3000] if result.stdout else "(empty)")
print("STDERR:", result.stderr[:3000] if result.stderr else "(empty)")
print("Exit:", result.returncode)
