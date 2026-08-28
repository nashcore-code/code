#!/usr/bin/env bash

artifact_root() {
  cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "ERROR: required command not found: $1" >&2
    exit 2
  }
}

require_new_directory() {
  local target="$1"
  if [[ -e "$target" ]]; then
    echo "ERROR: destination already exists: $target" >&2
    exit 2
  fi
  mkdir -p "$target"
}

line_count() {
  awk 'END { print NR }' "$1"
}

expect_lines() {
  local expected="$1" file="$2" actual
  actual="$(line_count "$file")"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: $file has $actual lines; expected $expected" >&2
    exit 1
  fi
}

expect_empty() {
  if [[ -s "$1" ]]; then
    echo "ERROR: expected an empty file: $1" >&2
    exit 1
  fi
}

check_sha256() {
  local expected="$1" file="$2" actual
  actual="$(python3 - "$file" <<'PY'
import hashlib
import pathlib
import sys
p = pathlib.Path(sys.argv[1])
h = hashlib.sha256()
with p.open('rb') as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
print(h.hexdigest())
PY
)"
  if [[ "$actual" != "$expected" ]]; then
    echo "ERROR: SHA-256 mismatch for $file" >&2
    echo "expected $expected" >&2
    echo "actual   $actual" >&2
    exit 1
  fi
}

