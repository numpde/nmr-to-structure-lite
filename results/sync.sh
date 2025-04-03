#!/bin/bash

# Exit on error, undefined variables, or failed pipeline commands
set -euo pipefail

# Ensure an argument is provided
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <lambda-hostname>"
  exit 1
fi

LAMBDA="$1"

# Resolve the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR"

# Ensure the destination directory exists
mkdir -p "$DEST_DIR"

echo "Syncing from $LAMBDA to $DEST_DIR every 100 seconds..."

while true; do
  echo "." && sleep 1
  echo "." && sleep 1
  echo "." && sleep 1

  echo "Starting sync at $(date)..."

  ssh ubuntu@"$LAMBDA" 'find /home/ubuntu/tmp/nmr-to-structure-lite/ -type f -mmin +0.1 -printf "%P\n"' | \
  rsync --archive --compress --verbose --progress --update \
        --files-from=- \
        ubuntu@"$LAMBDA":/home/ubuntu/tmp/nmr-to-structure-lite/ \
        "$DEST_DIR/"

  echo "Sync complete. Sleeping..."
  sleep 100
done
