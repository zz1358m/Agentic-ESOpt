#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
JITRL_WEBARENA="${JITRL_WEBARENA:-$ROOT/data/webarena/jitrl}"
PY="${PY:-python}"
HOST="${WEBARENA_HOST:-${1:-}}"

if [ -z "$HOST" ]; then
  echo "Usage: WEBARENA_HOST=<host> $0"
  echo "   or: $0 <host>"
  exit 2
fi

HOST="${HOST#http://}"
HOST="${HOST#https://}"
HOST="${HOST%/}"

SHOPPING_URL="http://${HOST}:7770"
SHOPPING_ADMIN_URL="http://${HOST}:7780/admin"
REDDIT_URL="http://${HOST}:9999"
GITLAB_URL="http://${HOST}:8023"
MAP_URL="http://${HOST}:3000"
WIKIPEDIA_URL="http://${HOST}:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
HOMEPAGE_URL="http://${HOST}:4399"

cat > "$JITRL_WEBARENA/env_setup.txt" <<EOF
export BASE_URL=${HOST}
export WA_SHOPPING=${SHOPPING_URL}
export WA_SHOPPING_ADMIN=${SHOPPING_ADMIN_URL}
export WA_REDDIT=${REDDIT_URL}
export WA_GITLAB=${GITLAB_URL}
export WA_MAP=${MAP_URL}
export WA_WIKIPEDIA=${WIKIPEDIA_URL}
export WA_HOMEPAGE=${HOMEPAGE_URL}
export SHOPPING=${SHOPPING_URL}
export SHOPPING_ADMIN=${SHOPPING_ADMIN_URL}
export REDDIT=${REDDIT_URL}
export GITLAB=${GITLAB_URL}
export MAP=${MAP_URL}
export WIKIPEDIA=${WIKIPEDIA_URL}
export HOMEPAGE=${HOMEPAGE_URL}
EOF

export BASE_URL="$HOST"
export WA_SHOPPING="$SHOPPING_URL"
export WA_SHOPPING_ADMIN="$SHOPPING_ADMIN_URL"
export WA_REDDIT="$REDDIT_URL"
export WA_GITLAB="$GITLAB_URL"
export WA_MAP="$MAP_URL"
export WA_WIKIPEDIA="$WIKIPEDIA_URL"
export WA_HOMEPAGE="$HOMEPAGE_URL"
export SHOPPING="$SHOPPING_URL"
export SHOPPING_ADMIN="$SHOPPING_ADMIN_URL"
export REDDIT="$REDDIT_URL"
export GITLAB="$GITLAB_URL"
export MAP="$MAP_URL"
export WIKIPEDIA="$WIKIPEDIA_URL"
export HOMEPAGE="$HOMEPAGE_URL"

mkdir -p "$JITRL_WEBARENA/config_files"
(
  cd "$JITRL_WEBARENA"
  "$PY" config_files_generation/generate_test_data.py
)

echo "JitRL WebArena env written to $JITRL_WEBARENA/env_setup.txt"
echo "Generated config files in $JITRL_WEBARENA/config_files"
echo "Host: $HOST"
