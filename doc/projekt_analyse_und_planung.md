# Breathwork Transcription Pipeline – Projektanalyse & Planungsdokument

> **Erstellt:** 2026-03-05  
> **Zweck:** Kontextdokument für die Weiterentwicklung der Pipeline (Speaker Diarization Integration)  
> **Status:** Analyse abgeschlossen, Implementierung ausstehend

---

## 1. Projektübersicht

**Ziel:** Automatische Transkription von Sprachaufnahmen (Voice Notes) aus einem Forschungsprojekt (Mikro-phänomenologische Interviews), verknüpft mit Video-Zeitstempeln aus JSON-Dateien – für die qualitative Analyse.

**Repo:** `breathwork-transcription`  
**Pipeline-Version:** 1.1.0  
**Betrieb:** 100% lokal, kein Internet, DSGVO/HIPAA-konform

---

## 2. Projektstruktur

```
breathwork-transcription/
├── pipeline/
│   ├── config.py              # Zentrale Konfiguration (Modell, Schwellenwerte, Pfade)
│   ├── run_pipeline.py        # Haupteinstiegspunkt – orchestriert alle Schritte
│   ├── preprocess_audio.py    # Audio-Vorverarbeitung via FFmpeg
│   ├── transcribe.py          # Whisper-Transkription (aktuell: openai-whisper)
│   ├── merge_outputs.py       # Output-Generierung (TXT, JSON, Reports)
│   └── utils.py               # Hilfsfunktionen (Pairing, Timestamps, Dateien)
├── doc/
│   └── projekt_analyse_und_planung.md   # Dieses Dokument
├── requirements.txt
├── README.md
└── README_cluster_for_new_users.md
```

---

## 3. Pipeline-Flow (Schritt für Schritt)

| Schritt | Modul | Was passiert |
|---------|-------|-------------|
| 1 | `utils.py` | `.wav`-Dateien mit `.json`-Timestamp-Dateien paaren |
| 2 | `preprocess_audio.py` | FFmpeg: Audio → 16kHz, Mono, 16-bit PCM + Loudness-Normalisierung |
| 3 | `transcribe.py` | Whisper transkribiert lokal (aktuell: `openai-whisper`) |
| 4 | `merge_outputs.py` | Individuelle Transkripte mit Video-Timestamps generieren |
| 5 | `merge_outputs.py` | Combined Transcript (TXT + JSON) erstellen |
| 6 | `merge_outputs.py` | Plain-Text-Transcript + Processing Report erstellen |
| 7 | `run_pipeline.py` | Cleanup der Zwischendateien |

---

## 4. Transkriptions-Engine (aktueller Stand)

```python
# pipeline/transcribe.py – aktuell
import whisper  # openai-whisper (Original)

model.transcribe(
    audio_file,
    language="en",           # ⚠️ Englisch – muss auf Deutsch umgestellt werden
    task="transcribe",
    temperature=0.0,         # Deterministisch/reproduzierbar
    verbose=False,
    word_timestamps=False    # ⚠️ Muss aktiviert werden (für Diarization)
)
```

---

## 5. Alle bestehenden Features (müssen ALLE erhalten bleiben)

| Feature | Details |
|---------|---------|
| **JSON-Timestamp-Verknüpfung** | `video_timestamp_sec` aus JSON → Video-Zeitstempel im Output |
| **Auto-Session-Erkennung** | Single Session oder Multiple Sessions (Batch) automatisch erkannt |
| **Audio-Preprocessing** | FFmpeg: 16kHz, Mono, 16-bit, Loudness-Normalisierung |
| **Parallele Vorverarbeitung** | Multiprocessing für Audio-Preprocessing |
| **Hallucination Detection** | Compression Ratio > 2,4 → Flag `hallucination_detected` |
| **Silence Detection** | No-Speech-Probability > 0,6 → Flag `silence_detected` |
| **Low Confidence Detection** | avg_logprob < -1,0 → Flag `low_confidence` |
| **Quality Flags** | Pro Segment + pro Recording aggregiert |
| **Orphaned Files** | Audio ohne JSON wird trotzdem transkribiert (markiert als orphaned) |
| **5 Output-Formate** | Individuelle TXT/JSON, Combined TXT/JSON, Plain Text, Processing Report |
| **Deterministisch** | Temperature 0.0 → reproduzierbare Ergebnisse |
| **Verbatim** | Filler Words (äh, hm, etc.) bleiben erhalten |
| **GPU Auto-Detection** | CUDA wenn verfügbar, sonst CPU |
| **Cluster-Support** | SLURM-kompatibel (GPUshortx86, RTX 6000) |
| **100% Lokal** | Keine Netzwerk-Anfragen, DSGVO/HIPAA-konform |

---

## 6. Input/Output Format

### Input
```
session_folder/
├── note_001.wav
├── note_001.json    →  { "video_timestamp_sec": 125.693 }
├── note_002.wav
└── note_002.json    →  { "video_timestamp_sec": 310.420 }
```

### Output (wird in `transcripts/` erstellt)
```
transcripts/
├── transcripts/
│   ├── note_001.txt       # Einzeltranskript mit Video-Timestamp-Header
│   ├── note_001.json      # Einzeltranskript mit Metadaten + Quality Metrics
│   └── ...
├── combined_transcript.txt    # Alle Transkripte zusammen, sortiert nach Timestamp
├── combined_transcript.json   # Maschinenlesbar, vollständige Metadaten
├── plain_text_transcript.txt  # Nur Text, keine Metadaten (für Textanalyse)
└── processing_report.txt      # Statistiken, Qualitätsmetriken, Systeminfo
```

---

## 7. Konfiguration (config.py) – wichtige Parameter

```python
WHISPER_MODEL = "small.en"          # ⚠️ Wird geändert (siehe Abschnitt 8)
LANGUAGE = "en"                     # ⚠️ Wird auf "de" geändert
TEMPERATURE = 0.0                   # Bleibt: deterministisch
ENABLE_QUALITY_CHECKS = True        # Bleibt
COMPRESSION_RATIO_THRESHOLD = 2.4  # Hallucination Detection
NO_SPEECH_THRESHOLD = 0.6          # Silence Detection
CONFIDENCE_THRESHOLD = -1.0        # Low Confidence Detection
TARGET_SAMPLE_RATE = 16000         # Whisper-optimiert
NORMALIZE_AUDIO = True             # Loudness-Normalisierung
NUM_PARALLEL_PROCESSES = None      # Alle CPU-Kerne für Preprocessing
SKIP_EXISTING = True               # Resume bei Unterbrechung
```

---

## 8. Geplante Änderungen & Entscheidungen

### 8.1 Engine-Wechsel: openai-whisper → faster-whisper

**Warum:**
- Das Speaker-Diarization-Projekt nutzt `faster-whisper`
- Beide Komponenten sollen dieselbe Engine verwenden
- faster-whisper ist ~3–4x schneller bei minimalem Genauigkeitsverlust
- Bessere Kontrolle über compute_type (float16, int8)

**Auswirkung auf Code:**
- `transcribe.py` muss komplett umgeschrieben werden
- `requirements.txt` muss angepasst werden (`openai-whisper` → `faster-whisper`)
- API-Unterschied: faster-whisper gibt Segmente als Generator zurück, nicht als Dict

### 8.2 Sprache: Englisch → Deutsch

**Warum:** Forschungsdaten sind ausschließlich auf Deutsch.

**Änderungen:**
```python
# config.py
LANGUAGE = "de"                    # Deutsch
WHISPER_MODEL = "large-v3"         # Multilinguales Modell (kein .en-Suffix!)
```

**Wichtig:** Die `.en`-Modelle (small.en, medium.en) sind English-only und können kein Deutsch. Es muss auf ein multilinguales Modell gewechselt werden. Empfehlung: `large-v3` für maximale Genauigkeit bei Forschungsdaten.

### 8.3 Word Timestamps aktivieren

**Warum:** Für Speaker Diarization zwingend notwendig.

**Hintergrund:** Ohne Word Timestamps wird ein ganzes Segment (Satz/Abschnitt) einem einzigen Sprecher zugeordnet. Wenn mitten im Segment der Sprecher wechselt (z.B. Interviewer fragt, Teilnehmer antwortet), wird alles falsch zugeordnet. Mit Word Timestamps kann der Sprecherwechsel auf Wortebene erkannt werden.

**Änderung:**
```python
# transcribe.py
word_timestamps=True   # War: False
```

### 8.4 Speaker Diarization als neues Feature

**Was es macht:** Erkennt automatisch, wer wann spricht, und ordnet jedem Textsegment einen Sprecher zu.

**Output-Format (geplant):**
```
[SPEAKER_00]: "Und was haben Sie dabei empfunden?"
[SPEAKER_01]: "Es war ein warmes Gefühl, fast wie..."
```

**Sprecher-Anzahl:**
- Normalfall: **2 Sprecher** (Interviewer + Teilnehmer)
- Ausnahmen möglich: **3–4 Sprecher**
- Automatische Zuordnung wer Interviewer/Teilnehmer ist: **nicht geplant** (zu komplex, manuell besser)

**Lokale Verarbeitung:** Muss vollständig offline funktionieren – keine HuggingFace-API-Calls zur Laufzeit. Ob ein einmaliger Token-Download akzeptabel ist, wird am Diarization-Projekt-Code geklärt.

---

## 9. Cluster-Setup (SLURM)

**Cluster:** HPC mit SLURM Workload Manager  
**GPU-Node:** `esi-svhpc107` mit RTX 6000  
**Partition:** `GPUshortx86`

### Conda Environment
```bash
conda create -n breathwork-py310 python=3.10 -y
conda activate breathwork-py310
pip install -r requirements.txt
conda install -y pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
```

### Job starten (GPU)
```bash
srun --partition=GPUshortx86 --nodelist=esi-svhpc107 --gpus=1 --pty $SHELL
module load conda
conda activate breathwork-py310
python pipeline/run_pipeline.py --input "input_file_path"
```

### Wichtig für x86-Architektur
Der SLURM-Job muss auf einem **x86-Knoten** ausgeführt werden:
```bash
sbatch --constraint=x86 mein_job.sh
# oder direkt:
srun --partition=GPUshortx86 ...
```

---

## 10. Nächste Schritte (Implementierungsplan – ausstehend)

- [ ] Speaker-Diarization-Projekt analysieren (Pfad noch nicht übergeben)
- [ ] Klären: Lokale Diarization ohne HuggingFace Token möglich?
- [ ] Implementierungsplan erstellen (Schritt für Schritt, laienverständlich)
- [ ] `transcribe.py` auf faster-whisper umschreiben
- [ ] `config.py` anpassen (Sprache, Modell, Word Timestamps)
- [ ] `requirements.txt` aktualisieren
- [ ] Diarization-Modul integrieren (neues `diarize.py`)
- [ ] `merge_outputs.py` erweitern (Speaker-Labels in alle Output-Formate)
- [ ] `run_pipeline.py` erweitern (Diarization als optionaler Schritt)
- [ ] Testen auf CPU (lokal) und GPU (Cluster)

---

## 11. Offene Fragen (noch ungeklärt)

- [ ] **HuggingFace Token / Lokale Diarization:** Welches Diarization-Modell wird genutzt? Läuft es komplett ohne Token? → Klärung nach Analyse des Diarization-Projekts
- [ ] **Modellgröße vs. Cluster-Ressourcen:** `large-v3` braucht ~10GB RAM + GPU-VRAM. Reicht die RTX 6000 (24GB VRAM)?

---

*Dieses Dokument wird nach Analyse des Diarization-Projekts ergänzt.*
