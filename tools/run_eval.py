#!/usr/bin/env python3
"""A/B-test one rule in an instruction file by running real agent tasks.

For each case: copy the fixture into a fresh temp dir, run `claude -p` there
with the rule file injected (arm "with") or with the rule removed (arm
"without"), save the full trace, then run the case's check script, which
decides pass/fail from what the agent actually did (files, logs), not from
what it said.

Usage:
  run_eval.py CASE_DIR [CASE_DIR ...] [--runs N] [--arms with,without]
              [--model M] [--launcher CMD] [--config-dir DIR] [--history FILE]

--model runs the agent on another Claude model (default: claude's own default).
Use a weaker model on the "without" arm to check that a case can catch the
mistake at all: if it fails there and passes on the current model, the case
works and the current model no longer needs the rule.

--launcher replaces the `claude` command, e.g. a wrapper script that points
Claude Code at another provider's Anthropic-compatible endpoint and picks the
model ("my-wrapper -m deepseek-flash"). The harness stays Claude Code; only the
model behind it changes. --config-dir names the Claude Code config dir that
launcher uses (default: $CLAUDE_CONFIG_DIR or ~/.claude); --history needs it.

--history FILE puts a prior conversation in front of the task: FILE is a session
template (one user/assistant message per line, cwd and sessionId written as
{{CWD}} and {{SID}}), installed as a session for the work dir and resumed, so
the task arrives as the next turn of a long session. Unknown models get a
200k-token auto-compact window; set CLAUDE_CODE_MAX_CONTEXT_TOKENS for longer.

case.json fields:
  target_file    file holding the rule, default ~/.claude/CLAUDE.md; a relative
                 path is read from the case dir
  rule_prefix    start of the rule's line (removed in "without"), or
  remove         list of {"line": prefix} / {"from": prefix, "to": prefix}
  extra_files    other instruction files given unchanged to both arms
  settings       merged over the default isolation settings, e.g.
                 {"outputStyle": "<name>"} to load an output style natively
  prompt         the task given to the agent
  allowed_tools  list passed to --allowedTools
  max_turns      optional, default 15
Layout: fixture/ becomes the agent's cwd; an optional setup.sh runs there first
(cwd = work dir, same env) for state a fixture folder cannot hold, such as a
git history; tools/ (fake CLIs) goes on PATH from a
dir outside cwd, so the agent cannot read the trap by browsing its folder; fakes
and check.sh share a state dir via $EVAL_STATE (also outside cwd).
The agent's final reply is saved to $EVAL_STATE/final.txt before check.sh runs.
check.sh runs with cwd = the work dir; exit 0 = pass, 2 = invalid (the run never
reached the situation the rule is about), anything else = fail. Its stdout is kept.
"""
import argparse, json, os, re, shlex, shutil, subprocess, sys, tempfile, time, uuid
from pathlib import Path

# The user's own rules are switched off (CLAUDE.md auto-load, hooks, output
# style) and fed back explicitly, so both arms differ only by the ablated rule.
SETTINGS = json.dumps({"claudeMdExcludes": ["**/CLAUDE.md"], "disableAllHooks": True,
                       "outputStyle": "default"})
DEFAULT_EXTRA = []


def ablate(text, remove):
    """Drop the rule from text. Each item: {"line": prefix} drops one line;
    {"from": prefix, "to": prefix} drops from the first line up to (not
    including) the second. Every prefix must match exactly one line."""
    lines = text.split("\n")
    def find(prefix):
        hits = [i for i, l in enumerate(lines) if l.startswith(prefix)]
        if len(hits) != 1:
            sys.exit(f"prefix must match exactly one line, matched {len(hits)}: {prefix!r}")
        return hits[0]
    drop = set()
    for item in remove:
        if "line" in item:
            drop.add(find(item["line"]))
        else:
            drop.update(range(find(item["from"]), find(item["to"])))
    return "\n".join(l for i, l in enumerate(lines) if i not in drop)


def resolve(case_dir, path):
    p = Path(os.path.expanduser(path))
    return p if p.is_absolute() else case_dir / p


def rules_text(case_dir, case, arm):
    """Instruction text the agent gets: the target file (ablated in the
    "without" arm) plus extra files unchanged, in that order."""
    target = resolve(case_dir, case.get("target_file", "~/.claude/CLAUDE.md"))
    remove = case.get("remove") or [{"line": case["rule_prefix"]}]
    text = Path(target).read_text(encoding="utf-8")
    ablated = ablate(text, remove)
    parts = [ablated if arm == "without" else text]
    for f in case.get("extra_files", DEFAULT_EXTRA):
        parts.append(resolve(case_dir, f).read_text(encoding="utf-8"))
    return "\n\n".join(parts)


def final_text(stream):
    final = ""
    for line in stream.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result":
            final = ev.get("result") or ""
    return final


def api_error(stream):
    """True when the run ended in an API error, at the first call or midway
    (e.g. the gateway quota running out). Hitting --max-turns is is_error too,
    but that is the agent's own outcome, so the check still decides it."""
    for line in stream.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "result" and ev.get("is_error"):
            return ev.get("subtype") != "error_max_turns"
    return not stream.strip()


def run_one(case_dir, case, arm, k, out_root, model=None, history=None,
            launcher="claude", config_dir=None):
    # realpath: macOS temp dirs sit behind /var -> /private/var, and an agent
    # writing to the /private path would miss the ./** permission rules.
    base = Path(tempfile.mkdtemp(prefix=f"{case_dir.name}-{arm}-{k}-")).resolve()
    work, tools, state = base / "work", base / ".t", base / ".s"
    shutil.copytree(case_dir / "fixture", work)
    state.mkdir()
    if (case_dir / "tools").exists():
        shutil.copytree(case_dir / "tools", tools)
        for p in tools.iterdir():
            p.chmod(0o755)
    if (case_dir / "setup.sh").exists():
        subprocess.run(["bash", str(case_dir.resolve() / "setup.sh")], cwd=work,
                       env=dict(os.environ, EVAL_STATE=str(state)), check=True,
                       capture_output=True)
    rules = base / "rules.md"
    rules.write_text(rules_text(case_dir, case, arm), encoding="utf-8")
    env = dict(os.environ, PATH=f"{tools}:{os.environ['PATH']}", EVAL_STATE=str(state))
    settings = dict(json.loads(SETTINGS), **case.get("settings", {}))
    cmd = shlex.split(launcher) + ["-p", case["prompt"], "--settings", json.dumps(settings),
           "--append-system-prompt-file", str(rules),
           "--allowedTools", ",".join(case["allowed_tools"]),
           "--max-turns", str(case.get("max_turns", 15)),
           "--output-format", "stream-json", "--verbose"]
    if model:
        cmd += ["--model", model]
    session_dir = None
    if history:
        sid = str(uuid.uuid4())
        config = Path(os.path.expanduser(config_dir or os.environ.get("CLAUDE_CONFIG_DIR", "~/.claude")))
        session_dir = config / "projects" / re.sub(r"[^A-Za-z0-9]", "-", str(work))
        session_dir.mkdir(parents=True, exist_ok=True)
        text = Path(history).read_text(encoding="utf-8")
        (session_dir / f"{sid}.jsonl").write_text(
            text.replace("{{CWD}}", json.dumps(str(work))[1:-1]).replace("{{SID}}", sid), encoding="utf-8")
        cmd += ["--resume", sid]
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=work, env=env, stdin=subprocess.DEVNULL,
                          capture_output=True, text=True, timeout=case.get("timeout", 600))
    run_dir = out_root / case_dir.name / f"{arm}-{k}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "trace.jsonl").write_text(proc.stdout, encoding="utf-8")
    if proc.stderr.strip():
        (run_dir / "stderr.txt").write_text(proc.stderr, encoding="utf-8")
    (state / "final.txt").write_text(final_text(proc.stdout), encoding="utf-8")
    chk = subprocess.run(["bash", str(case_dir.resolve() / "check.sh")], cwd=work,
                         env=env, capture_output=True, text=True)
    (run_dir / "check.txt").write_text(chk.stdout + chk.stderr, encoding="utf-8")
    (run_dir / "final.txt").write_text(final_text(proc.stdout), encoding="utf-8")
    for f in state.iterdir():
        if f.name != "final.txt" and f.is_file():
            shutil.copy(f, run_dir / f"state-{f.name}")
    shutil.rmtree(base, ignore_errors=True)
    if session_dir:
        shutil.rmtree(session_dir, ignore_errors=True)
    # An API failure leaves the fixture untouched, which some checks read as a
    # pass; a run whose agent errored out never reached the situation at all.
    if api_error(proc.stdout):
        return "INVALID", round(time.time() - t0), "agent run errored: " + final_text(proc.stdout)[:120]
    verdict = {0: "PASS", 2: "INVALID"}.get(chk.returncode, "FAIL")
    return verdict, round(time.time() - t0), chk.stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="+")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--arms", default="with,without")
    ap.add_argument("--out", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--history", default=None)
    ap.add_argument("--launcher", default="claude")
    ap.add_argument("--config-dir", default=None)
    a = ap.parse_args()
    out_root = Path(a.out or Path.cwd() / "results" / time.strftime("%Y%m%d-%H%M%S"))
    jobs = []
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=6) as ex:
        for c in a.cases:
            cd = Path(c)
            case = json.loads((cd / "case.json").read_text(encoding="utf-8"))
            for arm in a.arms.split(","):
                for k in range(1, a.runs + 1):
                    jobs.append((cd.name, arm, k, ex.submit(run_one, cd, case, arm, k, out_root, a.model, a.history,
                                                        a.launcher, a.config_dir)))
    summary = {}
    for name, arm, k, fut in jobs:
        verdict, secs, note = fut.result()
        summary.setdefault(name, {}).setdefault(arm, []).append(verdict)
        print(f"{name:32} {arm:8} run{k}  {verdict:7}  {secs:4}s  {note[:80]}")
    print()
    for name, arms in summary.items():
        print(name + "  " + "  ".join(
            f"{arm}: {v.count('PASS')}/{len(v) - v.count('INVALID')} pass"
            + (f" ({v.count('INVALID')} invalid)" if 'INVALID' in v else "")
            for arm, v in arms.items()))
    (out_root / "summary.json").parent.mkdir(parents=True, exist_ok=True)
    (out_root / "summary.json").write_text(json.dumps(dict(summary, _model=a.model or "default", _history=a.history), indent=1), encoding="utf-8")
    print(f"\ntraces: {out_root}")


if __name__ == "__main__":
    main()
