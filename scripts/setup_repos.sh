#!/usr/bin/env bash
#
# Clone breathwork-transcription and whisper-diarization as siblings.
# Run from the parent directory where both repos should live.
#
# Usage:
#   ./scripts/setup_repos.sh [parent_dir]
#
# Example (on cluster):
#   cd /gs/home/<your-username>
#   bash breathwork-transcription/scripts/setup_repos.sh .
#   # or clone breathwork-transcription first, then:
#   cd /gs/home/<your-username>
#   git clone https://github.com/Namsjain01/breathwork-transcription.git
#   bash breathwork-transcription/scripts/setup_repos.sh .
#
# If parent_dir is omitted, uses the directory containing this script's repo.

set -e

PARENT_DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PARENT_DIR"

BREATHWORK_REPO="${BREATHWORK_REPO_URL:-https://github.com/Namsjain01/breathwork-transcription.git}"
DIARIZATION_REPO="${DIARIZATION_REPO_URL:-https://github.com/MahmoudAshraf97/whisper-diarization.git}"

if [ -d "breathwork-transcription" ]; then
  echo "breathwork-transcription already exists in $PARENT_DIR"
else
  echo "Cloning breathwork-transcription into $PARENT_DIR ..."
  git clone "$BREATHWORK_REPO" breathwork-transcription
fi

if [ -d "whisper-diarization" ]; then
  echo "whisper-diarization already exists in $PARENT_DIR"
else
  echo "Cloning whisper-diarization into $PARENT_DIR ..."
  git clone "$DIARIZATION_REPO" whisper-diarization
fi

echo ""
echo "Done. Layout:"
echo "  $PARENT_DIR/"
echo "    breathwork-transcription/"
echo "    whisper-diarization/"
echo ""
echo "Next: cd breathwork-transcription && pip install -r requirements.txt"
echo "Then run: python pipeline/run_pipeline.py --input <path-to-session>"
