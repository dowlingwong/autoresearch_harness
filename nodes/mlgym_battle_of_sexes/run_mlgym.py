"""Fail closed until the MLGym container and workspace bridge is approved."""
import sys

print("MLGym execution unavailable: container access and workspace bridge are not configured.", file=sys.stderr)
raise SystemExit(2)
