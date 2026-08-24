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
# FFprobe verfügbar machen (optional)
# ==========================================
#
# imageio-ffmpeg liefert nur ffmpeg, kein ffprobe. Ein statischer Build
# (ffmpeg + ffprobe) kann nach $PROJECT_ROOT/bin entpackt werden.
#
# Ohne ffprobe läuft die Pipeline weiter: utils.get_audio_duration()
# liest die Dauer dann über soundfile aus dem WAV-Header.

if [[ -d "$PROJECT_ROOT/bin" ]]; then
    export PATH="$PROJECT_ROOT/bin:$PATH"
fi

if ! command -v ffprobe >/dev/null 2>&1; then
    echo "WARNUNG: ffprobe nicht im PATH."
    echo "         Audio-Dauern werden über soundfile bestimmt."
fi

# ==========================================
# CUDA-12-Libraries für CTranslate2
# ==========================================
#
# faster-whisper transkribiert über CTranslate2, nicht über PyTorch.
# CTranslate2 4.x lädt zur Laufzeit libcublas.so.12 und libcudnn_*.so.9.
# Das hier installierte Torch (cu118) bringt nur die CUDA-11-Varianten mit,
# deshalb liegen die CUDA-12-Libraries in einem eigenen Verzeichnis --
# getrennt von site-packages, damit sie sich nicht mit Torchs cu11-Libs mischen.
#
# Einmalig anlegen mit:
#   pip install --target "$PROJECT_ROOT/cuda12-libs" \
#       nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"

CUDA12_LIB_DIR="$PROJECT_ROOT/cuda12-libs"

if [[ ! -d "$CUDA12_LIB_DIR" ]]; then
    echo "ERROR: CUDA-12-Libraries für CTranslate2 nicht gefunden:"
    echo "       $CUDA12_LIB_DIR"
    echo
    echo "Einmalig installieren mit:"
    echo "  pip install --target \"$CUDA12_LIB_DIR\" \\"
    echo "      nvidia-cublas-cu12 \"nvidia-cudnn-cu12>=9,<10\""
    exit 1
fi

# Alle nvidia/<paket>/lib-Verzeichnisse einsammeln und voranstellen.
for libdir in "$CUDA12_LIB_DIR"/nvidia/*/lib; do
    [[ -d "$libdir" ]] || continue
    export LD_LIBRARY_PATH="$libdir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
done

if ! compgen -G "$CUDA12_LIB_DIR/nvidia/cublas/lib/libcublas.so.12*" >/dev/null; then
    echo "ERROR: libcublas.so.12 nicht in $CUDA12_LIB_DIR gefunden."
    echo "       Installation unvollständig -- siehe pip-Kommando oben."
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
echo "FFprobe:       $(command -v ffprobe || echo 'nicht gefunden (Fallback: soundfile)')"
echo "LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-nicht gesetzt}"
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
# CUDA-Library-Test für CTranslate2
# ==========================================
#
# Prüft dieselben Libraries, die CTranslate2 zur Laufzeit per dlopen lädt.
# Schlägt hier fehl, bevor stundenlang Audio vorverarbeitet wird.

echo
echo "CUDA-Library-Test (CTranslate2):"

"$PYTHON" - <<'PY'
import ctypes
import sys

import ctranslate2

print("CTranslate2-Version:", ctranslate2.__version__)
print("CUDA-Geräte:", ctranslate2.get_cuda_device_count())

# CTranslate2 >= 4.5 braucht cuBLAS aus CUDA 12 und cuDNN 9.
required = [
    "libcublas.so.12",
    "libcublasLt.so.12",
    "libcudnn.so.9",
    "libcudnn_ops.so.9",
    "libcudnn_cnn.so.9",
]

missing = []

for name in required:
    try:
        ctypes.CDLL(name)
        print(f"  OK      {name}")
    except OSError as error:
        print(f"  FEHLT   {name}  ({error})")
        missing.append(name)

if missing:
    print(
        "ERROR: CTranslate2 kann diese CUDA-Libraries nicht laden: "
        + ", ".join(missing),
        file=sys.stderr,
    )
    sys.exit(1)

if ctranslate2.get_cuda_device_count() < 1:
    print("ERROR: CTranslate2 sieht keine CUDA-GPU.", file=sys.stderr)
    sys.exit(1)
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