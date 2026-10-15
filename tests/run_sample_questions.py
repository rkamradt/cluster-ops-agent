#!/usr/bin/env python3
"""Run five sample questions against the live agent and report results + timings.

Usage:
    python tests/run_sample_questions.py [base_url]
    # default base_url: https://agent.kamradtfamily.net
"""
import sys
import time
import urllib.request
import urllib.parse
import json

BASE_URL = sys.argv[1] if len(sys.argv) > 1 else "https://agent.kamradtfamily.net"

QUESTIONS = [
    ("Basic pod listing",       "List all pods in the home namespace"),
    ("Cross-namespace health",  "Are there any unhealthy or crashlooping pods in any namespace?"),
    ("ArgoCD status",           "What is the ArgoCD sync and health status of the home application?"),
    ("Events",                  "Show me the most recent events in the k8s-dashboard namespace"),
    ("Diagnose broken pod",     "What is wrong with the broken-test pod in the cluster-ops-agent namespace?"),
]

SENSITIVE_PATTERNS = [
    "svc.cluster.local", "10.", "192.168.", "172.16.",
    "ghp_", "eyJ", "mongodb://", "password", "secret", "token=",
]

def ask(question: str) -> tuple[str, list[str], float]:
    """Stream one question; return (answer, progress_nodes, elapsed_seconds)."""
    url = f"{BASE_URL}/ask?question={urllib.parse.quote(question)}"
    start = time.time()
    nodes: list[str] = []
    answer = ""

    req = urllib.request.Request(url, headers={"Accept": "text/event-stream"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        event_type = ""
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                event_type = line[6:].strip()
            elif line.startswith("data:"):
                data = json.loads(line[5:].strip())
                if event_type == "progress":
                    nodes.append(data.get("node", "?"))
                elif event_type == "answer":
                    answer = data.get("content", "")
                elif event_type in ("done", "error"):
                    break

    return answer, nodes, time.time() - start


def check_sensitive(text: str) -> list[str]:
    return [p for p in SENSITIVE_PATTERNS if p.lower() in text.lower()]


def main():
    print(f"Target: {BASE_URL}\n{'═' * 70}")
    all_passed = True

    for i, (label, question) in enumerate(QUESTIONS, 1):
        print(f"\n[{i}/5] {label}")
        print(f"  Q: {question}")
        try:
            answer, nodes, elapsed = ask(question)
            leaks = check_sensitive(answer)
            status = "PASS" if not leaks else "WARN — sensitive data detected"

            print(f"  Nodes: {' → '.join(nodes) if nodes else '(none)'}")
            print(f"  Time:  {elapsed:.1f}s")
            print(f"  Status: {status}")
            if leaks:
                print(f"  Leaks: {leaks}")
                all_passed = False
            print(f"  Answer (first 300 chars):\n    {answer[:300].replace(chr(10), chr(10)+'    ')}")
        except Exception as exc:
            print(f"  ERROR: {exc}")
            all_passed = False

    print(f"\n{'═' * 70}")
    print("RESULT:", "ALL PASSED" if all_passed else "ISSUES FOUND — review output above")


if __name__ == "__main__":
    main()
