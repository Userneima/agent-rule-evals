#!/usr/bin/env python3
"""Cut a real Claude Code session into prior-context files for long-history runs.

Keeps the main chain's user and assistant messages (tool calls and results
included), drops images and sidechains, and cuts only right before a new user
request so every tool call keeps its result. Each cut is written as a session
template (cwd and sessionId left as placeholders, filled in per run) to
private/histories/<name>-<k>k.jsonl, plus a review sheet listing the user
requests it contains and any lines that look like secrets.

Usage: build_history.py SESSION.jsonl NAME TARGET_K [TARGET_K ...]
TARGET_K is the approximate prior length in thousands of tokens.
"""
import json, re, sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "private" / "histories"
CHARS_PER_TOKEN = 2.5  # rough for mixed Chinese/English/code; checked against real usage later
SECRET = re.compile(r"(sk-[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}|xox[bp]-[A-Za-z0-9-]{10,}|"
                    r"AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY|"
                    r"(?i:api[_-]?key|token|secret|password)\s*[=:]\s*\S{8,})")


def strip_images(content):
    if not isinstance(content, list):
        return content
    out = []
    for b in content:
        if isinstance(b, dict) and b.get("type") == "image":
            out.append({"type": "text", "text": "[image]"})
        elif isinstance(b, dict) and b.get("type") == "tool_result" and isinstance(b.get("content"), list):
            out.append(dict(b, content=strip_images(b["content"])))
        else:
            out.append(b)
    return out


def main():
    src, name, targets = sys.argv[1], sys.argv[2], sorted(int(t) for t in sys.argv[3:])
    msgs = []
    for line in open(src, encoding="utf-8"):
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") not in ("user", "assistant") or e.get("isSidechain") or e.get("isMeta"):
            continue
        m = dict(e["message"], content=strip_images(e["message"].get("content")))
        msgs.append({"type": e["type"], "message": m, "timestamp": e.get("timestamp")})
    # A new user request is a user message whose content is plain text.
    starts = [i for i, m in enumerate(msgs)
              if m["type"] == "user" and isinstance(m["message"]["content"], str)]
    sizes, total = [], 0
    for m in msgs:
        total += len(json.dumps(m["message"]["content"], ensure_ascii=False))
        sizes.append(total)
    OUT.mkdir(parents=True, exist_ok=True)
    for k in targets:
        want = k * 1000 * CHARS_PER_TOKEN
        cut = max([s for s in starts if s > 0 and sizes[s - 1] <= want] or [0])
        part = msgs[:cut]
        lines, prev = [], None
        for i, m in enumerate(part):
            uid = f"h{i:05d}-0000-4000-8000-000000000000"
            lines.append({"parentUuid": prev, "isSidechain": False, "type": m["type"],
                          "message": m["message"], "uuid": uid, "timestamp": m["timestamp"],
                          "userType": "external", "cwd": "{{CWD}}", "sessionId": "{{SID}}"})
            prev = uid
        f = OUT / f"{name}-{k}k.jsonl"
        f.write_text("\n".join(json.dumps(l, ensure_ascii=False) for l in lines) + "\n", encoding="utf-8")
        text = "\n".join(json.dumps(m["message"]["content"], ensure_ascii=False) for m in part)
        asks = [m["message"]["content"] for m in part
                if m["type"] == "user" and isinstance(m["message"]["content"], str)]
        hits = sorted({h.group(0)[:40] for h in SECRET.finditer(text)})
        review = [f"# {f.name}", f"约 {len(text) / CHARS_PER_TOKEN / 1000:.0f}k token，{len(part)} 条消息，{len(asks)} 次提问",
                  "", "## 疑似敏感内容", *(f"- `{h}`" for h in hits or ["（无）"]), "", "## 用户的提问"]
        review += [f"{n}. " + a.strip().replace("\n", " ")[:200] for n, a in enumerate(asks, 1)]
        (OUT / f"{name}-{k}k.review.md").write_text("\n".join(review) + "\n", encoding="utf-8")
        print(f.name, f"~{len(text) / CHARS_PER_TOKEN / 1000:.0f}k tokens", len(asks), "asks", len(hits), "secret-like")


if __name__ == "__main__":
    main()
