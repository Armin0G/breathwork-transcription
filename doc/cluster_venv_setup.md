# Cluster-Setup: venv statt Conda

Anleitung für den ESI-Cluster (`esi-svhpc107`), nachdem Conda dort nicht mehr
nutzbar ist. Alle Kommandos werden **auf dem Cluster** ausgeführt, auf dem
Login-Node reicht aus (keine GPU nötig, nur `pip`).

Projektpfad: `/cs/home/goffina/breathwork-transcription`
(derselbe Ort ist über `/mnt/hpc/home/goffina/...` gemountet)

---

## Warum das nötig ist

`faster-whisper` rechnet nicht über PyTorch, sondern über **CTranslate2**.
CTranslate2 4.5+ lädt zur Laufzeit per `dlopen`:

- `libcublas.so.12` / `libcublasLt.so.12` (cuBLAS aus CUDA **12**)
- `libcudnn*.so.9` (cuDNN **9**)

Im Conda-Env kamen diese Libraries aus dem Env selbst. Die venv enthält
`torch 2.6.0+cu118` — das liefert nur die CUDA-**11**-Varianten
(`libcublas.so.11`). Deshalb der Fehler:

```
Error transcribing …: Library libcublas.so.12 is not found or cannot be loaded
```

`torch.cuda.is_available()` ist dabei `True` und das Whisper-Modell lädt auch
noch — der Fehler kommt erst beim ersten echten GPU-Rechenschritt.

Die CUDA-12-Libraries kommen deshalb in ein **eigenes Verzeichnis** neben der
venv, nicht in `site-packages`. So mischen sie sich nicht mit den cu11-Libs von
Torch. Torch lädt seine eigenen Libraries über RPATH und ist von
`LD_LIBRARY_PATH` nicht betroffen.

---

## Schritt 1 — Code aktualisieren

```bash
cd /cs/home/goffina/breathwork-transcription
git pull
```

## Schritt 2 — CUDA-12-Libraries installieren (einmalig)

```bash
cd /cs/home/goffina/breathwork-transcription
source .venv/bin/activate

pip install --target ./cuda12-libs \
    nvidia-cublas-cu12 "nvidia-cudnn-cu12>=9,<10"
```

Das legt ca. 1–2 GB unter `cuda12-libs/nvidia/{cublas,cudnn}/lib/` ab.
`cuda12-libs/` steht in `.gitignore`.

**Kontrolle:**

```bash
ls cuda12-libs/nvidia/cublas/lib/libcublas.so.12*
ls cuda12-libs/nvidia/cudnn/lib/libcudnn_ops.so.9*
```

Beide Kommandos müssen je eine Datei ausgeben.

### Falls eine ältere CTranslate2-Version installiert ist

```bash
pip show ctranslate2 | grep Version
```

- **≥ 4.5** → cuDNN 9, also das Kommando oben (Normalfall)
- **4.0–4.4** → braucht cuDNN 8:
  `pip install --target ./cuda12-libs nvidia-cublas-cu12 "nvidia-cudnn-cu12>=8,<9"`
  In dem Fall zusätzlich in `run_pipeline_list.sh` die Liste `required` im
  Abschnitt „CUDA-Library-Test" von `.so.9` auf `.so.8` anpassen.

## Schritt 3 — ffprobe (optional)

`imageio-ffmpeg` liefert nur `ffmpeg`, kein `ffprobe`. Die Pipeline liest die
Audiodauer inzwischen bevorzugt über `soundfile` aus dem WAV-Header, läuft also
auch ohne `ffprobe`. Wer beide Binaries will:

```bash
cd /cs/home/goffina/breathwork-transcription
mkdir -p bin
cd bin
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xf ffmpeg-release-amd64-static.tar.xz --strip-components=1 \
    --wildcards '*/ffmpeg' '*/ffprobe'
rm ffmpeg-release-amd64-static.tar.xz
chmod +x ffmpeg ffprobe
```

`run_pipeline_list.sh` nimmt `$PROJECT_ROOT/bin` automatisch in den PATH, wenn
das Verzeichnis existiert. `bin/` steht in `.gitignore`.

Prüfen, dass `soundfile` da ist (sonst fällt der Duration-Fallback aus):

```bash
python -c "import soundfile; print(soundfile.__version__)"
```

## Schritt 4 — Setup verifizieren (ohne Job)

Auf dem **Login-Node** reicht der Library-Test; der GPU-Teil braucht eine GPU.

```bash
cd /cs/home/goffina/breathwork-transcription
source .venv/bin/activate

export LD_LIBRARY_PATH="$PWD/cuda12-libs/nvidia/cublas/lib:$PWD/cuda12-libs/nvidia/cudnn/lib:$LD_LIBRARY_PATH"

python - <<'PY'
import ctypes

for name in ("libcublas.so.12", "libcublasLt.so.12",
             "libcudnn.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"):
    try:
        ctypes.CDLL(name)
        print("OK     ", name)
    except OSError as error:
        print("FEHLT  ", name, "->", error)
PY
```

Alle fünf Zeilen müssen `OK` zeigen.

## Schritt 5 — Job starten

```bash
cd /cs/home/goffina/breathwork-transcription
sbatch run_pipeline_list.sh
```

Das Job-Skript macht das Setzen von `LD_LIBRARY_PATH` selbst und bricht **vor**
der Audio-Vorverarbeitung ab, wenn etwas fehlt. Im Log erscheint jetzt
zusätzlich:

```
FFprobe:       …
LD_LIBRARY_PATH: …

CUDA-Library-Test (CTranslate2):
CTranslate2-Version: …
CUDA-Geräte: 1
  OK      libcublas.so.12
  …
```

## Schritt 6 — Erfolgskontrolle

Im Output-Log sollte stehen:

```
✓ Successfully transcribed 14/14 files
```

Nicht `1/14`. Und die Transkripte sollten Text enthalten, nicht `(0 chars, 0 words)`.

---

## Troubleshooting

| Symptom | Ursache | Fix |
|---|---|---|
| `Library libcublas.so.12 is not found` | `cuda12-libs` fehlt oder `LD_LIBRARY_PATH` nicht gesetzt | Schritt 2, dann Schritt 4 |
| `libcudnn_ops.so.9: cannot open shared object file` | cuDNN-8-Wheel installiert | `pip install --target ./cuda12-libs --upgrade "nvidia-cudnn-cu12>=9,<10"` |
| `CUDA-Geräte: 0` im Job-Log | Job ohne GPU-Allocation gestartet | `#SBATCH --gpus=1` prüfen |
| `(0 chars, 0 words)` bei allen Dateien | VAD filtert alles weg, Audio evtl. stumm | Input-WAV mit `ffplay`/`soundfile` prüfen |
| `Warning: … has no 'video_timestamp_sec'` | JSON-Sidecar ohne Timestamp | Transkript wird trotzdem geschrieben, mit `[VIDEO TIMESTAMP: UNKNOWN]` — siehe unten |

### Offener Punkt: fehlende Video-Timestamps

Bei Dateien vom Typ `audio_at_frame_<N>_…` (und bei
`audio_at_00-00-05_…` aus Session `2RX900/PMR`) enthält die JSON-Sidecar-Datei
keinen Schlüssel `video_timestamp_sec`. Das hat die Pipeline bisher mit

```
TypeError: unsupported type for timedelta seconds component: NoneType
```

abstürzen lassen. Jetzt wird stattdessen `UNKNOWN` geschrieben und eine Warnung
ausgegeben; die Session läuft durch.

Ob sich der Timestamp aus der Frame-Nummer und der Videoframerate rekonstruieren
lässt, ist noch offen. Dafür wird der Inhalt einer solchen JSON-Datei gebraucht:

```bash
cat "/cs/projects/HEBznlReset/ZNLRESET-DATA/VIDEO/1W2ML9/PMR/video_recording_2025-09-16T13_14_47/audio_at_frame_0_16-15-26-32-364407.json"
```

---

## Rückbau

Falls das Setup rückgängig gemacht werden soll:

```bash
rm -rf /cs/home/goffina/breathwork-transcription/cuda12-libs
rm -rf /cs/home/goffina/breathwork-transcription/bin
```

Die venv selbst bleibt davon unberührt.
