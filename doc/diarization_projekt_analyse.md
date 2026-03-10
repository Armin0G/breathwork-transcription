# Whisper-Diarization Projekt – Analyse & Referenzdokument

> **Erstellt:** 2026-03-05
> **Zweck:** Referenzdokument für die Integration des Diarization-Projekts in die Breathwork Transcription Pipeline
> **Quell-Repo:** `whisper-diarization` (MahmoudAshraf97/whisper-diarization)
> **Status:** Analyse abgeschlossen, Integration ausstehend

---

## 1. Was ist dieses Projekt?

Ein Speaker-Diarization-System, das **faster-whisper** für Transkription mit **NVIDIA NeMo** für Sprechererkennung kombiniert. Es beantwortet die Frage: **„Wer hat wann was gesagt?"**

Das Projekt kombiniert mehrere spezialisierte Modelle zu einer Pipeline:
- Sprache → Text (Whisper)
- Text → präzise Wort-Zeitstempel (CTC Forced Aligner)
- Audio → Sprecher-Segmente (NeMo MSDD)
- Wörter + Sprecher → fertiges Transkript mit Sprecher-Labels

---

## 2. Projektstruktur

```
whisper-diarization/
├── diarize.py                          # Haupt-Script (sequenziell, empfohlen)
├── diarize_parallel.py                 # Parallel-Variante (experimentell)
├── helpers.py                          # Hilfsfunktionen (Mapping, Output-Generierung)
├── diarization/
│   ├── __init__.py                     # Exportiert MSDDDiarizer
│   └── msdd/
│       ├── msdd.py                     # MSDD Diarizer Klasse (NeMo Wrapper)
│       └── diar_infer_telephonic.yaml  # NeMo Konfiguration
├── requirements.txt
├── constraints.txt                     # numpy<2, indic-numtowords
└── README.md
```

---

## 3. Pipeline-Flow (Schritt für Schritt)

| Schritt | Was passiert | Modell / Tool | Zweck |
|---------|-------------|---------------|-------|
| 1 | **Vocal Isolation** | `Demucs` (htdemucs) | Musik/Hintergrundgeräusche vom Sprachsignal trennen |
| 2 | **Transkription** | `faster-whisper` (BatchedInferencePipeline) | Audio → Text mit groben Zeitstempeln |
| 3 | **Forced Alignment** | `ctc-forced-aligner` | Zeitstempel auf Wortebene präzise korrigieren |
| 4 | **VAD** | `MarbleNet` (via NeMo) | Stille erkennen und ausschließen |
| 5 | **Speaker Embeddings** | `TitaNet Large` (via NeMo) | Stimm-Fingerabdrücke pro Segment extrahieren |
| 6 | **MSDD Diarization** | `diar_msdd_telephonic` (NeMo) | Wörter den Sprechern zuordnen |
| 7 | **Punctuation Restoration** | `kredor/punctuate-all` | Satzzeichen wiederherstellen für besseres Realignment |
| 8 | **Realignment** | `helpers.py` | Sprecherwechsel an Satzgrenzen ausrichten |
| 9 | **Output** | `helpers.py` | TXT + SRT Datei schreiben |

---

## 4. Verwendete Modelle im Detail

### 4.1 faster-whisper (Transkription)
- **Standard-Modell:** `medium.en` (wird für Integration auf `large-v3` geändert)
- **Modus:** `BatchedInferencePipeline` für schnelle Inferenz
- **Compute Type:** `float16` auf GPU, `int8` auf CPU
- **Batch Size:** 8 (Standard), reduzierbar bei wenig VRAM

### 4.2 Demucs – htdemucs (Vocal Isolation)
- **Zweck:** Trennt Gesang/Sprache von Musik und Hintergrundgeräuschen
- **Für Interviews:** ⚠️ Meist unnötig – verlangsamt die Pipeline
- **Flag:** `--no-stem` deaktiviert diesen Schritt
- **Empfehlung für Integration:** Standardmäßig deaktivieren (`stemming=False`)

### 4.3 ctc-forced-aligner (Forced Alignment)
- **Zweck:** Korrigiert Whisper-Zeitstempel auf Wortebene
- **Warum nötig:** Whisper-Timestamps haben oft leichte Verschiebungen; präzise Wort-Timestamps sind für korrekte Sprecher-Zuordnung essenziell
- **Input:** Transkript-Text + Audio-Waveform
- **Output:** Wort-genaue Start/End-Zeitstempel

### 4.4 MarbleNet (VAD – Voice Activity Detection)
- **Modell:** `vad_multilingual_marblenet`
- **Zweck:** Erkennt Sprachaktivität, filtert Stille heraus
- **Multilingual:** ✅ Funktioniert auf Deutsch
- **Quelle:** NVIDIA NeMo (automatischer Download, dann lokal gecacht)

### 4.5 TitaNet Large (Speaker Embeddings)
- **Modell:** `titanet_large`
- **Zweck:** Extrahiert einen „Stimm-Fingerabdruck" für jedes Audio-Segment
- **Multiscale:** Analysiert Fenster von 0,5s bis 1,5s für robuste Embeddings
- **Quelle:** NVIDIA NeMo (automatischer Download, dann lokal gecacht)

### 4.6 MSDD – Multi-Scale Diarization Decoder (Kern-Diarization)
- **Modell:** `diar_msdd_telephonic`
- **Zweck:** Ordnet Sprecher-Embeddings den Zeitabschnitten zu
- **Optimiert für:** Telefongespräche / Interviews (2–8 Sprecher)
- **Max. Sprecher:** 8 (konfigurierbar)
- **Quelle:** NVIDIA NeMo (automatischer Download, dann lokal gecacht)

### 4.7 deepmultilingualpunctuation (Satzzeichen)
- **Modell:** `kredor/punctuate-all`
- **Zweck:** Stellt Satzzeichen im Transkript wieder her
- **Unterstützte Sprachen inkl. Deutsch:** ✅ `de` ist in `punct_model_langs` enthalten
- **Quelle:** HuggingFace (öffentlich, kein Token nötig)

---

## 5. Abhängigkeiten & Lokale Verarbeitung

### requirements.txt
```
nemo_toolkit[asr]>=2.3.0
nltk
faster-whisper>=1.1.0
git+https://github.com/MahmoudAshraf97/demucs.git
git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git
```

### constraints.txt
```
numpy<2
indic-numtowords @ git+https://github.com/AI4Bharat/indic-numtowords.git
```

### Lokale Verarbeitung – Übersicht

| Modell | Quelle | Token nötig? | Nach Download lokal? |
|--------|--------|-------------|---------------------|
| faster-whisper | HuggingFace | ❌ Nein | ✅ Ja |
| Demucs | GitHub | ❌ Nein | ✅ Ja |
| ctc-forced-aligner | GitHub | ❌ Nein | ✅ Ja |
| MarbleNet (VAD) | NVIDIA NeMo | ❌ Nein | ✅ Ja (`~/.cache/torch/NeMo/`) |
| TitaNet Large | NVIDIA NeMo | ❌ Nein | ✅ Ja (`~/.cache/torch/NeMo/`) |
| diar_msdd_telephonic | NVIDIA NeMo | ❌ Nein | ✅ Ja (`~/.cache/torch/NeMo/`) |
| kredor/punctuate-all | HuggingFace | ❌ Nein | ✅ Ja (`~/.cache/huggingface/`) |

> ✅ **Fazit: Kein HuggingFace Token nötig. Nach dem ersten Download läuft alles vollständig offline.**

---

## 6. Output-Format des Diarization-Projekts

### TXT-Datei
```
Speaker 0: "Und was haben Sie dabei empfunden?"

Speaker 1: "Es war ein warmes Gefühl, fast wie eine Wärme die sich ausbreitet."

Speaker 0: "Können Sie das genauer beschreiben?"
```

### SRT-Datei (Untertitel-Format)
```
1
00:00:01,240 --> 00:00:04,820
Speaker 0: Und was haben Sie dabei empfunden?

2
00:00:05,100 --> 00:00:09,340
Speaker 1: Es war ein warmes Gefühl, fast wie eine Wärme die sich ausbreitet.
```

### Was fehlt für die Integration ins Hauptprojekt
- ❌ Kein JSON-Output (muss neu gebaut werden)
- ❌ Keine Video-Timestamp-Verknüpfung
- ❌ Keine Quality Flags
- ❌ Keine Session-Struktur / Batch-Verarbeitung
- ❌ Kein Processing Report

---

## 7. Deutsch-Kompatibilität

| Feature | Deutsch unterstützt? | Details |
|---------|---------------------|---------|
| Transkription | ✅ Ja | `large-v3` Modell, `language="de"` |
| Forced Alignment | ✅ Ja | `langs_to_iso["de"] = "ger"` in helpers.py |
| VAD (MarbleNet) | ✅ Ja | Multilingual-Modell |
| Speaker Embeddings (TitaNet) | ✅ Ja | Sprachunabhängig (Stimm-Merkmale) |
| MSDD Diarization | ✅ Ja | Sprachunabhängig (Timing-basiert) |
| Punctuation Restoration | ✅ Ja | `"de"` ist in `punct_model_langs` enthalten |

---

## 8. Konfiguration (NeMo YAML)

Relevante Parameter aus `diar_infer_telephonic.yaml`:

```yaml
# VAD Parameter
vad:
  model_path: vad_multilingual_marblenet
  parameters:
    onset: 0.1          # Schwellenwert für Sprachbeginn
    offset: 0.1         # Schwellenwert für Sprachende
    pad_onset: 0.1      # Puffer vor Sprachsegment
    min_duration_off: 0.2  # Minimale Pause zwischen Segmenten

# Speaker Embeddings
speaker_embeddings:
  model_path: titanet_large
  parameters:
    window_length_in_sec: [1.5, 1.25, 1.0, 0.75, 0.5]  # Multiscale-Analyse
    shift_length_in_sec:  [0.75, 0.625, 0.5, 0.375, 0.25]

# Clustering
clustering:
  parameters:
    oracle_num_speakers: False  # Sprecher-Anzahl automatisch erkennen
    max_num_speakers: 8         # Maximum (für uns: auf 4 reduzierbar)

# MSDD
msdd_model:
  model_path: diar_msdd_telephonic
  parameters:
    sigmoid_threshold: [0.7]    # Schwellenwert für Sprecher-Überlappung
```

---

## 9. Wichtige Beobachtungen für die Integration

### ✅ Vorteile / Gut geeignet
- MSDD-Modell ist für **Telefon-/Interview-Gespräche** optimiert → passt perfekt
- **Deutsch vollständig unterstützt** in allen Komponenten
- **Kein HuggingFace Token** nötig
- **Forced Alignment** sorgt für präzise Wort-Timestamps → bessere Sprecher-Zuordnung
- Sprecher-Anzahl wird **automatisch erkannt** (kein manuelles Setzen nötig)

### ⚠️ Anpassungsbedarf für Integration
- **Demucs deaktivieren:** Für reine Sprach-Interviews unnötig, verlangsamt Pipeline
- **Kein JSON-Output:** Muss für Hauptprojekt neu implementiert werden
- **Kein Batch/Session-Support:** Diarization-Projekt verarbeitet einzelne Dateien
- **Output-Pfade:** Festes `input/`- und `output/`-Verzeichnis → muss flexibel werden
- **Modell-Name:** Standard `medium.en` → muss auf `large-v3` geändert werden
- **Sprache:** Standard `en` → muss auf `de` geändert werden

### 🔧 Was neu gebaut werden muss
- `diarize.py` Logik als **Modul/Klasse** kapseln (nicht als Standalone-Script)
- **JSON-Output** mit Speaker-Labels für jedes Segment
- **Integration in `run_pipeline.py`** als optionaler Schritt nach Transkription
- **Speaker-Labels in alle Output-Formate** einbauen (TXT, Combined TXT, Combined JSON)
- **`config.py` erweitern** um Diarization-Parameter

---

## 10. Geplante Architektur nach Integration

```
breathwork-transcription/pipeline/
├── config.py              # + Diarization-Einstellungen
├── run_pipeline.py        # + Diarization als optionaler Schritt
├── preprocess_audio.py    # Unverändert
├── transcribe.py          # Umgeschrieben auf faster-whisper
├── diarize.py             # NEU – Diarization-Modul (aus whisper-diarization extrahiert)
├── merge_outputs.py       # + Speaker-Labels in alle Outputs
└── utils.py               # Minimal angepasst
```

### Geplanter erweiterter Output (mit Diarization)

**combined_transcript.txt:**
```
────────────────────────────────────────────────────────────────────────────────
ANNOTATION #1
VIDEO TIMESTAMP: 00:02:05.693 (125.693 seconds)
AUDIO FILE: note1.wav
DURATION: 6.2 seconds
────────────────────────────────────────────────────────────────────────────────

[SPEAKER_00]: Und was haben Sie dabei empfunden?

[SPEAKER_01]: Es war ein warmes Gefühl, fast wie eine Wärme die sich ausbreitet.
```

**combined_transcript.json (Erweiterung):**
```json
{
  "annotations": [
    {
      "id": 1,
      "video_timestamp_sec": 125.693,
      "transcription": "Und was haben Sie dabei empfunden? Es war ein warmes Gefühl...",
      "diarization": [
        {
          "speaker": "SPEAKER_00",
          "start_time_ms": 0,
          "end_time_ms": 2340,
          "text": "Und was haben Sie dabei empfunden?"
        },
        {
          "speaker": "SPEAKER_01",
          "start_time_ms": 2800,
          "end_time_ms": 6200,
          "text": "Es war ein warmes Gefühl, fast wie eine Wärme die sich ausbreitet."
        }
      ],
      "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
      "quality_flags": []
    }
  ]
}
```

---

## 11. Ressourcen-Anforderungen

| Komponente | RAM | VRAM (GPU) |
|-----------|-----|-----------|
| faster-whisper large-v3 | ~4 GB | ~6 GB |
| Demucs (wenn aktiv) | ~4 GB | ~3 GB |
| NeMo (VAD + TitaNet + MSDD) | ~4 GB | ~4 GB |
| **Gesamt (ohne Demucs)** | **~8 GB** | **~10 GB** |
| **Gesamt (mit Demucs)** | **~12 GB** | **~13 GB** |

> RTX 6000 (24 GB VRAM) auf dem Cluster ist ausreichend. ✅

---

## 12. Nächste Schritte

- [ ] Implementierungsplan erstellen (Schritt für Schritt)
- [ ] `transcribe.py` auf faster-whisper umschreiben + `word_timestamps=True`
- [ ] `config.py` um Diarization-Parameter erweitern
- [ ] Neues `diarize.py` Modul für Hauptprojekt schreiben
- [ ] `merge_outputs.py` um Speaker-Labels erweitern
- [ ] `run_pipeline.py` um Diarization-Schritt erweitern
- [ ] `requirements.txt` zusammenführen
- [ ] Testen auf CPU (lokal) und GPU (Cluster, SLURM)

---

*Dieses Dokument wird während der Implementierung aktualisiert.*
