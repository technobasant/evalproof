#!/usr/bin/env bash
# Run the whole pytest suite in Linux containers: other Python versions, a musl libc (Alpine)
# and an emulated architecture. It checks portability before CI does; the host's own
# `uv run pytest` stays the quick check.
#
#   scripts/test-in-docker.sh                      all targets
#   scripts/test-in-docker.sh --only py313,alpine313
#   scripts/test-in-docker.sh --list
#   scripts/test-in-docker.sh --target 'py313-bookworm python:3.13-slim-bookworm'
#
# What a target runs:
#   * The source is a snapshot of the working tree: the files git tracks (staged new files
#     included, files you deleted left out), with your uncommitted edits. Untracked files,
#     .git and .venv are not in it. `git add` a new file for it to be tested.
#   * The snapshot is mounted read-only; the container copies it to a writable /work.
#   * As root the container installs git and uv. It then drops to an unprivileged user and
#     runs `uv sync --frozen` and the tests there, because root bypasses permission checks and
#     some tests rely on them. The drop is a `su` to a user the container creates, in the same
#     container: git and uv stay installed, and the same code path works on Debian and on
#     Alpine (busybox has `su` and `adduser`, but its `setpriv` cannot change user).
#   * Each target's full log is a file in the log directory, which is printed first.
#
# Needs Docker, git, tar, and network access (images, apt/apk, PyPI). Works with bash 3.2.
# The snapshot stays in the log directory, next to the logs, so you can see what the
# containers saw. Delete the directory when you are done.

set -euo pipefail

# name, image and an optional --platform value, one target per line
DEFAULT_TARGETS='py311 python:3.11-slim
py312 python:3.12-slim
py313 python:3.13-slim
py314 python:3.14-slim
alpine313 python:3.13-alpine
amd64-313 python:3.13-slim linux/amd64'

usage() {
  # The comment block at the top of this file is the help text.
  sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

die() {
  echo "test-in-docker: $*" >&2
  exit 2
}

only=""
log_dir=""
list_only=0
custom_targets=""
while [ $# -gt 0 ]; do
  case "$1" in
    --only)
      [ $# -ge 2 ] || die "--only needs NAME[,NAME]"
      only="$2"
      shift 2
      ;;
    --target)
      [ $# -ge 2 ] || die "--target needs 'NAME IMAGE [PLATFORM]'"
      custom_targets="${custom_targets}${2}"$'\n'
      shift 2
      ;;
    --log-dir)
      [ $# -ge 2 ] || die "--log-dir needs a directory"
      log_dir="$2"
      shift 2
      ;;
    --list)
      list_only=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      die "unknown argument: $1 (try --help)"
      ;;
  esac
done

if [ -n "$custom_targets" ]; then
  targets="$custom_targets"
else
  targets="$DEFAULT_TARGETS"
fi

if [ "$list_only" -eq 1 ]; then
  printf '%s\n' "$targets" | while read -r name image platform; do
    [ -n "$name" ] || continue
    printf '%-12s %-24s %s\n' "$name" "$image" "${platform:-native}"
  done
  exit 0
fi

# Which targets run: all of them, or the ones --only names (an unknown name is an error)
selected=""
all_names=""
while read -r name image platform; do
  [ -n "$name" ] || continue
  [ -n "$image" ] || die "target '$name' has no image"
  all_names="$all_names $name"
  if [ -n "$only" ]; then
    case ",$only," in
      *",$name,"*) selected="${selected}${name} ${image} ${platform:-}"$'\n' ;;
    esac
  else
    selected="${selected}${name} ${image} ${platform:-}"$'\n'
  fi
done <<EOF
$targets
EOF
if [ -n "$only" ]; then
  old_ifs="$IFS"
  IFS=,
  for want in $only; do
    case " $all_names " in
      *" $want "*) ;;
      *) die "no target named '$want' (targets:$all_names)" ;;
    esac
  done
  IFS="$old_ifs"
fi
[ -n "$selected" ] || die "no targets selected"

command -v docker >/dev/null 2>&1 || die "docker is not installed"
docker info >/dev/null 2>&1 || die "the docker daemon is not running"
repo=$(git rev-parse --show-toplevel 2>/dev/null) || die "run this inside the evalproof git repository"
cd "$repo"

# The same uv as the release workflow, unless UV_VERSION says otherwise
if [ -z "${UV_VERSION:-}" ]; then
  UV_VERSION=$(sed -n 's/^  UV_VERSION: "\(.*\)"$/\1/p' .github/workflows/release.yml | head -n 1)
fi
[ -n "$UV_VERSION" ] || die "could not read UV_VERSION from .github/workflows/release.yml; set it in the environment"

if [ -n "$log_dir" ]; then
  mkdir -p "$log_dir"
else
  log_dir=$(mktemp -d "${TMPDIR:-/tmp}/evalproof-docker-tests.XXXXXX")
fi
echo "logs and source snapshot: $log_dir"

# Snapshot: tracked files that still exist, with their working-tree content. COPYFILE_DISABLE
# stops macOS tar from adding ._ metadata files.
snapshot="$log_dir/source"
mkdir -p "$snapshot"
git ls-files -z | while IFS= read -r -d '' file; do
  if [ -e "$file" ] || [ -L "$file" ]; then
    printf '%s\0' "$file"
  fi
done | COPYFILE_DISABLE=1 tar --null -T - -cf - | tar -xf - -C "$snapshot"
echo "snapshot: $(find "$snapshot" -type f | wc -l | tr -d ' ') files"

# The script the containers run: first as root (packages, a user), then, through su, as that
# user (sync and tests). Plain POSIX sh, because Alpine has no bash.
inner="$log_dir/inner"
mkdir -p "$inner"
cat >"$inner/inner.sh" <<'INNER'
set -eu
phase="${1:-root}"

case "$phase" in
  root)
    echo "==> python $(python -c 'import platform; print(platform.python_version())')"
    echo "==> machine $(uname -m), $(. /etc/os-release && echo "$PRETTY_NAME")"

    if command -v apt-get >/dev/null 2>&1; then
      export DEBIAN_FRONTEND=noninteractive
      apt-get update -qq
      apt-get install -y -qq --no-install-recommends git >/dev/null
    elif command -v apk >/dev/null 2>&1; then
      apk add --no-cache git >/dev/null
    else
      echo "neither apt-get nor apk: cannot install git" >&2
      exit 2
    fi
    echo "==> $(git --version)"

    python -m pip install --quiet --disable-pip-version-check --root-user-action=ignore \
      "uv==${UV_VERSION}"
    echo "==> $(uv --version)"

    # Debian has useradd; Alpine (busybox) has adduser
    if command -v useradd >/dev/null 2>&1; then
      useradd --create-home --uid 10001 tester
    else
      adduser -D -u 10001 tester
    fi
    mkdir -p /work
    cp -a /src/. /work/
    chown -R tester /work

    exec su -s /bin/sh tester -c "sh /inner/inner.sh user"
    ;;
  user)
    export HOME=/home/tester
    cd /work
    echo "==> running as uid $(id -u)"
    if [ "$(id -u)" = 0 ]; then
      echo "still root: the privilege drop failed" >&2
      exit 2
    fi
    # The container's own Python, never one uv would download
    UV_PYTHON=$(command -v python)
    export UV_PYTHON UV_PYTHON_DOWNLOADS=never UV_NO_PROGRESS=1
    uv sync --frozen
    exec uv run --frozen python -m pytest -p no:cacheprovider -o addopts="" -q \
      -W error::ResourceWarning
    ;;
  *)
    echo "unknown phase: $phase" >&2
    exit 2
    ;;
esac
INNER

# Run the selected targets one after another
results=""
failed=0
while read -r name image platform; do
  [ -n "$name" ] || continue
  log="$log_dir/$name.log"
  plat_args=()
  if [ -n "$platform" ]; then
    plat_args=(--platform "$platform")
  fi
  echo
  echo "==> $name: $image${platform:+ ($platform)}"
  echo "    log: $log"
  start=$(date +%s)
  # --init: the tests kill process groups and then look for the dead processes. Without an init
  # as process 1, an orphaned grandchild stays a zombie that nothing reaps, and the tests
  # report it as still alive.
  if docker run --rm --init \
    ${plat_args[@]+"${plat_args[@]}"} \
    -e "UV_VERSION=$UV_VERSION" \
    -v "$snapshot:/src:ro" \
    -v "$inner:/inner:ro" \
    "$image" sh /inner/inner.sh root </dev/null >"$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  seconds=$(($(date +%s) - start))

  python_version=$(sed -n 's/^==> python //p' "$log" | head -n 1)
  summary=$(grep -E '^(=+ )?(([0-9]+ (passed|failed|skipped|errors?|xfailed|xpassed|deselected|warnings?))|no tests ran).* in [0-9.]+s' "$log" | tail -n 1 || true)
  count() { printf '%s\n' "$summary" | grep -oE "[0-9]+ $1" | grep -oE '^[0-9]+' | head -n 1 || true; }
  passed=$(count passed)
  skipped=$(count skipped)
  failed_n=$(count failed)
  errors_n=$(count 'errors?')
  bad=$((${failed_n:-0} + ${errors_n:-0}))
  if [ "$rc" -eq 0 ]; then
    result=PASS
  elif [ -n "$summary" ]; then
    result=FAIL
    failed=1
  else
    result=ERROR # no pytest summary: the setup failed before or during the tests
    failed=1
  fi
  results="${results}${name}|${python_version:--}|${result}|${passed:-0}/${bad}/${skipped:-0}|${seconds}"$'\n'
  echo "    $result in ${seconds}s (docker exit $rc)"
  if [ "$result" != PASS ]; then
    echo "    last lines of the log:"
    tail -n 15 "$log" | sed 's/^/    | /'
  fi
done <<EOF
$selected
EOF

echo
printf '%-12s %-9s %-7s %-22s %s\n' target python result 'passed/failed/skipped' seconds
printf '%s' "$results" | while IFS='|' read -r name python_version result counts seconds; do
  printf '%-12s %-9s %-7s %-22s %s\n' "$name" "$python_version" "$result" "$counts" "$seconds"
done
echo
echo "logs: $log_dir"
exit "$failed"
