#!/usr/bin/env bash

#SBATCH --partition=GPUshortx86
#SBATCH --nodelist=esi-svhpc107
#SBATCH --gpus=1
#SBATCH --job-name=tsc_pmr
#SBATCH --error=./log/error_tsc_pmr_%j.log
#SBATCH --output=./log/output_tsc_pmr_%j.log
#SBATCH --time=04:00:00

set -Eeuo pipefail

PROJECT_ROOT="/cs/home/goffina/breathwork-transcription"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON="$VENV_DIR/bin/python"
LISTFILE="$PROJECT_ROOT/session_list_pmr.txt"

# ==========================================
# Projektverzeichnis
# ==========================================

cd "$PROJECT_ROOT" || {
    echo "ERROR: Projektverzeichnis nicht gefunden:"
    echo "       $PROJECT_ROOT"
    exit 1
}

mkdir -p "$PROJECT_ROOT/log"

# ==========================================
# Virtuelle Umgebung prüfen und aktivieren
# ==========================================

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python-Interpreter nicht gefunden:"
    echo "       $PYTHON"
    exit 1
fi

source "$VENV_DIR/bin/activate"

# Die .venv soll bei allen Subprozessen vorne im PATH stehen.
export PATH="$VENV_DIR/bin:$PATH"

# ==========================================
# FFmpeg verfügbar machen
# ==========================================

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Kein System-FFmpeg im PATH gefunden."
    echo "Suche FFmpeg über imageio-ffmpeg ..."

    FFMPEG_BIN="$("$PYTHON" -c \
        'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"

    if [[ ! -x "$FFMPEG_BIN" ]]; then
        echo "ERROR: FFmpeg-Binary nicht gefunden oder nicht ausführbar:"
        echo "       $FFMPEG_BIN"
        exit 1
    fi

    # imageio-ffmpeg nennt die Binary beispielsweise:
    # ffmpeg-linux-x86_64-v7.0.2
    #
    # Die Pipeline erwartet jedoch das Kommando:
    # ffmpeg
    ln -sfn "$FFMPEG_BIN" "$VENV_DIR/bin/ffmpeg"

    # Nach dem Erstellen des Symlinks .venv/bin erneut priorisieren.
    export PATH="$VENV_DIR/bin:$PATH"
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "ERROR: FFmpeg ist weiterhin nicht im PATH."
    echo "PATH=$PATH"
    exit 1
fi

# ==========================================
# Jobinformationen
# ==========================================

echo "=========================================="
echo "Job gestartet: $(date)"
echo "Hostname:      $(hostname)"
echo "Projekt:       $PROJECT_ROOT"
echo "Python:        $("$PYTHON" --version)"
echo "Interpreter:   $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-nicht gesetzt}"
echo "FFmpeg:        $(command -v ffmpeg)"
echo "=========================================="

echo
echo "FFmpeg-Version:"
ffmpeg -version | head -n 1

echo
echo "GPU-Status:"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "ERROR: nvidia-smi nicht gefunden."
    exit 1
fi

nvidia-smi

# ==========================================
# PyTorch-/CUDA-Test
# ==========================================

echo
echo "PyTorch-Test:"

"$PYTHON" - <<'PY'
import sys
import torch

print("PyTorch-Version:", torch.__version__)
print("PyTorch-CUDA:", torch.version.cuda)
print("CUDA verfügbar:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print(
        "ERROR: PyTorch kann keine CUDA-GPU verwenden.",
        file=sys.stderr,
    )
    sys.exit(1)

print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU-Speicher:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
    "GB",
)
PY

# ==========================================
# Listen-Datei prüfen
# ==========================================

if [[ ! -f "$LISTFILE" ]]; then
    echo "ERROR: Listen-Datei nicht gefunden:"
    echo "       $LISTFILE"
    exit 1
fi

# Anzahl gültiger Einträge bestimmen
total="$("$PYTHON" - "$LISTFILE" <<'PY'
import sys

listfile = sys.argv[1]
count = 0

with open(listfile, encoding="utf-8") as file:
    for line in file:
        line = line.strip()

        if not line:
            continue

        if line.startswith("#"):
            continue

        count += 1

print(count)
PY
)"

if [[ "$total" -eq 0 ]]; then
    echo "ERROR: Die Listen-Datei enthält keine gültigen Einträge:"
    echo "       $LISTFILE"
    exit 1
fi

echo
echo "Anzahl Sessions: $total"

n=0
failed=0

# ==========================================
# Sessions verarbeiten
# ==========================================

while IFS= read -r path || [[ -n "$path" ]]; do
    # Whitespace am Anfang und Ende entfernen
    path="${path#"${path%%[![:space:]]*}"}"
    path="${path%"${path##*[![:space:]]}"}"

    # Leere Zeilen und Kommentare überspringen
    [[ -z "$path" ]] && continue
    [[ "$path" == \#* ]] && continue

    n=$((n + 1))

    echo
    echo "=========================================="
    echo "Processing $n/$total: $path"
    echo "=========================================="

    if "$PYTHON" pipeline/run_pipeline.py \
        --input "$path" \
        --no-diarization < /dev/null
    then
        echo "Successfully processed: $path"
    else
        exitcode=$?
        echo "WARNING: Failed with exit code $exitcode"
        failed=$((failed + 1))
    fi

done < "$LISTFILE"

# ==========================================
# Zusammenfassung
# ==========================================

successful=$((n - failed))

echo
echo "=========================================="
echo "Job finished at $(date)"
echo "Successfully processed: $successful/$n sessions"

if [[ "$failed" -gt 0 ]]; then
    echo "Failed: $failed sessions"
fi

echo "=========================================="

# Der Job endet mit Fehlerstatus, wenn mindestens
# eine Session fehlgeschlagen ist.
if [[ "$failed" -gt 0 ]]; then
    exit 1
fi

exit 0