#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_dataset.py
---------------
Create high-quality SFT datasets from multiple folders containing student projects.

Modes:
  - file : target-file synthesis (spec + context -> full file content)
  - diff : patch examples from git history (commit message -> unified diff)

Features:
- Project discovery (markers, file-count thresholds, per-extension balance)
- Deduplication (normalization + hash)
- Secret stripping (.env, API keys, TOKENS) from context/targets
- Length control (approx token estimator + byte limit with SNIP)
- Balanced and random sampling with seed
- Train/val split and optional shuffle
- Output format: CodeLlama [INST] or "chat" (messages list)

Example:
  python make_dataset.py \
      --roots "/path/SubjectA" "/path/SubjectB" \
      --out ./out \
      --modes file diff \
      --samples-per-project 40 \
      --format inst \
      --val-frac 0.05
"""

from __future__ import annotations
import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import hashlib
from pathlib import Path
from typing import Iterable, List, Dict

# ------------------------------- Constants -------------------------------

DEFAULT_INCLUDE_EXT = {
    ".py", ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini",
    ".js", ".ts", ".tsx", ".jsx",
    ".java", ".kt", ".scala",
    ".c", ".cc", ".cpp", ".h", ".hpp",
    ".go", ".rs",
    ".php", ".rb",
    ".sql",
    ".html", ".css",
    ".sh", ".bash", ".zsh",
    # Special files without extension handled below (Dockerfile/Makefile)
}

DEFAULT_EXCLUDE_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".venv", "venv",
    "node_modules", "build", "dist", ".idea", ".vscode", ".DS_Store", ".ruff_cache",
    ".gradle", "target", ".next", ".nuxt", ".turbo", ".parcel-cache"
}

# Heuristic tokens for "hot" files
HOT_TOKENS = ("main", "app", "server", "route", "controller", "service",
              "model", "utils", "helper", "config", "settings", "views",
              "api", "handler")

# System prompt (English)
SYSTEM_PROMPT_EN = (
    "You are a helpful programming assistant. Follow the instructions exactly.\n"
    "When creating or modifying files, respond with either the complete file contents "
    "or as a unified diff. Keep the code buildable and consistent with the project's style."
)

# CodeLlama [INST] wrapper
def wrap_inst_en(instruction: str, output: str) -> str:
    return (
        "<s>[INST] <<SYS>>\n" + SYSTEM_PROMPT_EN + "\n<</SYS>>\n" +
        instruction.strip() + "\n[/INST]\n" +
        output.strip() + "</s>"
    )

# ------------------------------- Utilities -------------------------------

SECRET_PATTERNS = [
    re.compile(r"(?i)api[_-]?key\s*=\s*['\"][A-Za-z0-9_\-]{12,}['\"]"),
    re.compile(r"(?i)secret\s*=\s*['\"][A-Za-z0-9/_\-\.\+]{12,}['\"]"),
    re.compile(r"(?i)bearer\s+[A-Za-z0-9\-\._~\+\/]+=*"),
    re.compile(r"(?i)ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)AIza[0-9A-Za-z\-_]{35}"),  # Google API key
    re.compile(r"(?i)sk-[A-Za-z0-9]{20,}"),     # generic pattern
]

ENV_LIKE = {".env", ".env.local", ".env.development", ".env.production"}

def sanitize_secrets(text: str) -> str:
    s = text
    for pat in SECRET_PATTERNS:
        s = pat.sub("<REDACTED>", s)
    return s

def is_binary(p: Path) -> bool:
    try:
        with open(p, "rb") as f:
            chunk = f.read(2048)
        return b"\0" in chunk
    except Exception:
        return True

def should_include_file(p: Path, include_ext: set) -> bool:
    if p.is_dir():
        return False
    if any(part in DEFAULT_EXCLUDE_DIRS for part in p.parts):
        return False
    if is_binary(p):
        return False
    base = p.name
    ext = p.suffix.lower()
    if base in {"Dockerfile", "Makefile"}:
        return True
    if base in ENV_LIKE:
        return False
    return ext in include_ext

def normalize_for_hash(s: str) -> str:
    s = re.sub(r"\r\n?", "\n", s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip().lower()

def text_hash(s: str) -> str:
    return hashlib.sha256(normalize_for_hash(s).encode("utf-8", errors="ignore")).hexdigest()

def approx_token_count(s: str) -> int:
    # very rough approximation: 1 token ~ 4 latin characters
    return max(1, len(s) // 4)

def read_text_safe(p: Path) -> str:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""

def trim_with_snip(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8", errors="ignore")
    if len(b) <= max_bytes:
        return s
    half = max_bytes // 2
    head = b[:half].decode("utf-8", errors="ignore")
    tail = b[-half:].decode("utf-8", errors="ignore")
    return head + "\n...<SNIP>...\n" + tail

def file_tree_str(root: Path, include_ext: set, max_files: int = 400) -> str:
    lines = []
    count = 0
    for p in sorted(root.rglob("*")):
        if any(part in DEFAULT_EXCLUDE_DIRS for part in p.parts):
            continue
        rel = p.relative_to(root)
        if p.is_dir():
            lines.append(str(rel) + "/")
            continue
        if should_include_file(p, include_ext):
            lines.append(str(rel))
            count += 1
            if count >= max_files:
                break
    return "\n".join(lines)

def neighbors_for_target(target: Path, project_root: Path, include_ext: set, k: int = 4) -> List[Path]:
    neigh = []
    # same directory
    for p in sorted(target.parent.glob("*")):
        if p == target or p.is_dir():
            continue
        if should_include_file(p, include_ext):
            neigh.append(p)
    # one level up: often key files
    for p in sorted(project_root.glob("*.py")):
        if should_include_file(p, include_ext):
            neigh.append(p)
    random.shuffle(neigh)
    return neigh[:k]

def project_candidates(root: Path, include_ext: set, min_files: int = 8) -> List[Path]:
    candidates = []
    markers = {"pyproject.toml", "package.json", "requirements.txt", "pom.xml", "setup.py"}
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        # prune
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_EXCLUDE_DIRS]
        files = [dp / f for f in filenames]
        included = [p for p in files if should_include_file(p, include_ext)]
        if any((dp / m).exists() for m in markers) or len(included) >= min_files:
            candidates.append(dp)
            dirnames[:] = []  # don't descend deeper
    return sorted(set(candidates))

def git_available() -> bool:
    return shutil.which("git") is not None

def is_git_repo(path: Path) -> bool:
    return (path / ".git").exists()

def run_git(args: List[str], cwd: Path) -> str:
    try:
        out = subprocess.check_output(["git"] + args, cwd=str(cwd), stderr=subprocess.STDOUT, text=True)
        return out
    except subprocess.CalledProcessError:
        return ""

# ------------------------------- Sample Builders -------------------------------

def score_hot(p: Path) -> int:
    name = p.name.lower()
    return sum(tok in name for tok in HOT_TOKENS)

def build_file_samples(project_root: Path,
                       include_ext: set,
                       max_file_bytes: int,
                       max_relevant_files: int,
                       samples_per_project: int,
                       max_tokens_instruction: int,
                       max_tokens_output: int,
                       dedup_set: set) -> List[dict]:
    files = [p for p in project_root.rglob("*") if should_include_file(p, include_ext)]
    # balance: group by extension so it's not all .py
    by_ext: Dict[str, List[Path]] = {}
    for p in files:
        by_ext.setdefault(p.suffix.lower() or p.name, []).append(p)
    for lst in by_ext.values():
        lst.sort(key=score_hot, reverse=True)
        random.shuffle(lst)
    # round-robin merge by extension
    merged = []
    changed = True
    keys = list(by_ext.keys())
    idx = {k: 0 for k in keys}
    while changed and len(merged) < samples_per_project * 4:
        changed = False
        for k in keys:
            l = by_ext[k]
            i = idx[k]
            if i < len(l):
                merged.append(l[i]); idx[k] += 1; changed = True
    chosen = merged[: max(12, samples_per_project * 3)]

    out = []
    tree = file_tree_str(project_root, include_ext)
    for target in chosen:
        if len(out) >= samples_per_project:
            break
        target_rel = str(target.relative_to(project_root))
        raw = read_text_safe(target)
        if not raw.strip():
            continue
        # trim + sanitize
        target_content = sanitize_secrets(trim_with_snip(raw, max_file_bytes))
        # skip overly large/unhelpful files
        if approx_token_count(target_content) > max_tokens_output:
            continue

        neigh = neighbors_for_target(target, project_root, include_ext, k=max_relevant_files)
        ctx_parts = [f"Project root: {project_root.name}",
                     "Project structure:\n" + tree,
                     "Relevant files:"]
        for n in neigh:
            rel = n.relative_to(project_root)
            ctx_parts.append(f"\n<file: {rel}>\n{sanitize_secrets(trim_with_snip(read_text_safe(n), max_file_bytes))}")

        instruction = (
            f"Create or update the file `{target_rel}` based on the project structure and existing conventions.\n"
            f"Output ONLY the complete contents of `{target_rel}`.\n\n" + "\n".join(ctx_parts)
        )

        # deduplicate on (instruction, output)
        dedup_key = text_hash(instruction + "\n###\n" + target_content)
        if dedup_key in dedup_set:
            continue
        dedup_set.add(dedup_key)

        if approx_token_count(instruction) > max_tokens_instruction:
            continue

        sample = {
            "project": str(project_root),
            "path": target_rel,
            "mode": "file",
        }
        sample["inst"] = instruction
        sample["out"] = target_content
        out.append(sample)

    return out

def build_diff_samples(project_root: Path,
                       max_commits: int,
                       max_diff_bytes: int,
                       max_tokens_instruction: int,
                       dedup_set: set) -> List[dict]:
    if not (git_available() and is_git_repo(project_root)):
        return []
    log = run_git(["log", f"--pretty=%H%x09%s", f"-n{max_commits}"], cwd=project_root)
    samples = []
    for line in log.splitlines():
        if "\t" not in line:
            continue
        sha, msg = line.split("\t", 1)
        diff = run_git(["show", "--format=", "--unified=0", sha], cwd=project_root)
        if not diff.strip():
            continue
        diff = sanitize_secrets(trim_with_snip(diff, max_diff_bytes))
        instruction = (
            "Apply the following change request (extracted from the commit message) to the project.\n"
            "Respond as a unified diff applied from the project root.\n\n"
            f"Change request: {msg}\n"
        )
        if approx_token_count(instruction) > max_tokens_instruction:
            continue
        dedup_key = text_hash(instruction + "\n###\n" + diff)
        if dedup_key in dedup_set:
            continue
        dedup_set.add(dedup_key)
        samples.append({
            "project": str(project_root),
            "mode": "diff",
            "commit": sha,
            "inst": instruction,
            "out": diff
        })
    return samples

# ------------------------------- Serialization -------------------------------

def to_inst_text(s: dict) -> str:
    """Convert a sample to CodeLlama [INST] text."""
    return wrap_inst_en(s["inst"], s["out"])

def to_chat_obj(s: dict) -> dict:
    """Convert a sample to 'chat' format (messages list)."""
    return {
        "project": s.get("project"),
        "mode": s.get("mode"),
        "path": s.get("path"),
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT_EN},
            {"role": "user", "content": s["inst"]},
            {"role": "assistant", "content": s["out"]},
        ],
    }

# ------------------------------- CLI -------------------------------

def parse_args():
    ap = argparse.ArgumentParser(
        prog="make_dataset.py",
        description="Generate an SFT dataset from student projects.",
    )
    ap.add_argument("--roots", nargs="+", required=True,
                    help="Paths to folders with projects (recursive).")
    ap.add_argument("--out", required=True,
                    help="Output folder (writes train.jsonl and val.jsonl).")
    ap.add_argument("--modes", nargs="+", default=["file"], choices=["file", "diff"],
                    help="Which sample types to build: file, diff.")
    ap.add_argument("--samples-per-project", type=int, default=30,
                    help="Max number of file samples per project.")
    ap.add_argument("--max-file-bytes", type=int, default=4000,
                    help="Trim file contents to this many bytes (head+tail with <SNIP>).")
    ap.add_argument("--max-diff-bytes", type=int, default=8000,
                    help="Trim unified diff to this many bytes.")
    ap.add_argument("--max-relevant-files", type=int, default=4,
                    help="Number of neighboring files to include as context.")
    ap.add_argument("--max-projects", type=int, default=999999,
                    help="Limit on total number of projects.")
    ap.add_argument("--seed", type=int, default=42, help="Seed for randomness.")
    ap.add_argument("--format", choices=["inst", "chat"], default="inst",
                    help="Output format: inst = CodeLlama [INST] text; chat = messages JSON.")
    ap.add_argument("--val-frac", type=float, default=0.02,
                    help="Validation split fraction (e.g., 0.02 = 2%).")
    ap.add_argument("--no-shuffle", action="store_true",
                    help="Do not shuffle samples before the train/val split.")
    ap.add_argument("--max-tok-inst", type=int, default=1600,
                    help="Rough token limit for the instruction.")
    ap.add_argument("--max-tok-out", type=int, default=2000,
                    help="Rough token limit for the target output (file content).")
    ap.add_argument("--include-ext", nargs="*", default=None,
                    help="Additional extensions (e.g., .py .md .json).")
    ap.add_argument("--max-commits", type=int, default=60,
                    help="Max number of commits to use in diff mode.")
    return ap.parse_args()

# ------------------------------- Main -------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)

    # include extensions
    if args.include_ext:
        extra = {e if e.startswith(".") else "." + e for e in args.include_ext}
    else:
        extra = set()
    include_ext = set(DEFAULT_INCLUDE_EXT) | extra

    roots = [Path(p).expanduser().resolve() for p in args.roots]
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = args.modes
    samples_per_project = args.samples_per_project
    max_file_bytes = args.max_file_bytes
    max_diff_bytes = args.max_diff_bytes
    max_rel = args.max_relevant_files
    max_projects = args.max_projects
    seed = args.seed
    out_format = args.format
    val_frac = args.val_frac
    do_shuffle = not args.no_shuffle
    max_tok_inst = args.max_tok_inst
    max_tok_out  = args.max_tok_out
    max_commits  = args.max_commits

    random.seed(seed)

    # discover projects
    all_projects: List[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[WARN] Does not exist: {root}", file=sys.stderr)
            continue
        found = project_candidates(root, include_ext)
        print(f"[INFO] {root}: found {len(found)} projects")
        all_projects.extend(found)

    # dedupe & cap
    all_projects = sorted(set(all_projects), key=lambda p: (len(str(p)), str(p)))[:max_projects]

    # build samples
    dedup: set = set()
    samples: List[dict] = []

    for proj in all_projects:
        print(f"[INFO] Processing project: {proj}")
        try:
            if "file" in modes:
                fs = build_file_samples(
                    project_root=proj,
                    include_ext=include_ext,
                    max_file_bytes=max_file_bytes,
                    max_relevant_files=max_rel,
                    samples_per_project=samples_per_project,
                    max_tokens_instruction=max_tok_inst,
                    max_tokens_output=max_tok_out,
                    dedup_set=dedup,
                )
                samples.extend(fs)

            if "diff" in modes:
                ds = build_diff_samples(
                    project_root=proj,
                    max_commits=max_commits,
                    max_diff_bytes=max_diff_bytes,
                    max_tokens_instruction=max_tok_inst,
                    dedup_set=dedup,
                )
                samples.extend(ds)

        except Exception as e:
            print(f"[WARN] Skipping project: {proj}\n{e}", file=sys.stderr)
            continue

    if do_shuffle:
        random.shuffle(samples)

    # convert to output format
    out_train = out_dir / "train.jsonl"
    out_val   = out_dir / "val.jsonl"
    n_total = len(samples)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    train_samples = samples[:n_train]
    val_samples = samples[n_train:]

    def write_jsonl(path: Path, it: Iterable[dict]):
        with path.open("w", encoding="utf-8") as f:
            for s in it:
                if out_format == "inst":
                    payload = {"text": to_inst_text(s), "mode": s["mode"], "project": s.get("project"), "path": s.get("path")}
                else:
                    payload = to_chat_obj(s)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    write_jsonl(out_train, train_samples)
    write_jsonl(out_val, val_samples)

    print(f"[DONE] train: {n_train}  val: {n_val}  -> {out_dir}")

if __name__ == "__main__":
    main()
