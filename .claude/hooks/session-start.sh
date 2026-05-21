#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

echo '{"async": true, "asyncTimeout": 300000}'

# Install project Python dependencies (best-effort — some packages may be incompatible)
if [ -f "$CLAUDE_PROJECT_DIR/requirements.txt" ]; then
  pip install -q -r "$CLAUDE_PROJECT_DIR/requirements.txt" || true
fi

# Install skills globally so they are available in every project
npx --yes skills add --global K-Dense-AI/scientific-agent-skills 2>/dev/null
npx --yes skills add --global amanning3390/hermeshub 2>/dev/null
