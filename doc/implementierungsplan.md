# Implementierungsplan: Speaker Diarization Integration

> **Erstellt:** 2026-03-05
> **Zweck:** Schritt-für-Schritt Anleitung zur Integration von Speaker Diarization in die Breathwork Transcription Pipeline
> **Ziel:** Alle bestehenden Features bleiben erhalten, Speaker Diarization wird als neues optionales Feature hinzugefügt
> **Status:** Bereit zur Umsetzung

---

## Überblick: Was wird gemacht?

Wir nehmen das bestehende Hauptprojekt (`breathwork-transcription`) und erweitern es in **6 Phasen**:

| Phase | Was passiert | Dateien betroffen |
|-------|-------------|-------------------|
| 1 | Abhängigkeiten aktualisieren | `requirements.txt` |
| 2 | Konfiguration erweitern | `config.py` |
| 3 | Transkription auf faster-whisper umstellen | `transcribe.py` |
| 4 | Neues Diarization-Modul erstellen | `diarize.py` (NEU) |
| 5 | Outputs um Speaker-Labels erweitern | `merge_outputs.py` |
| 6 | Pipeline orchestrieren | `run_pipeline.py` |

**Wichtige Grundregel:** Jede Phase ist in sich abgeschlossen und testbar. Wir gehen erst zur nächsten Phase, wenn die aktuelle funktioniert.

---

## Phase 1: Abhängigkeiten aktualisieren (`requirements.txt`)

### Was ist das Problem?

Das Hauptprojekt nutzt aktuell `openai-whisper` (das Original von OpenAI). Das Diarization-Projekt nutzt `faster-whisper`. Wir müssen beide auf eine gemeinsame Engine bringen – `faster-whisper` – und zusätzlich alle Diarization-Bibliotheken hinzufügen.

### Was wird geändert?

**Alte `requirements.txt` (Hauptprojekt):**
```
openai-whisper>=20231117
ffmpeg-python>=0.2.0
numpy>=1.20.0
pandas>=1.3.0
...
torch>=2.5.0
```

**Neue `requirements.txt` (nach Integration):**
```
# ============================================================
# TRANSKRIPTION (faster-whisper ersetzt openai-whisper)
# ============================================================
faster-whisper>=1.1.0

# ============================================================
# DIARIZATION (neu hinzugefügt)
# ============================================================
nemo_toolkit[asr]>=2.3.0
nltk
git+https://github.com/MahmoudAshraf97/demucs.git
git+https://github.com/oliverguhr/deepmultilingualpunctuation.git
git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# ============================================================
# AUDIO / ALLGEMEIN (bleibt)
# ============================================================
ffmpeg-python>=0.2.0
numpy<2
torch>=2.5.0
torchaudio>=2.5.0
tqdm>=4.60.0
pyyaml>=6.0

# ============================================================
# CONSTRAINTS (wichtig für Kompatibilität)
# ============================================================
# numpy muss <2 sein wegen NeMo-Kompatibilität
# indic-numtowords wird von NeMo benötigt:
# pip install git+https://github.com/AI4Bharat/indic-numtowords.git
```

### Warum `numpy<2`?

NeMo (das Diarization-Framework von NVIDIA) ist noch nicht vollständig mit NumPy 2.x kompatibel. Wenn wir NumPy 2.x installieren, gibt es Fehler beim Import von NeMo. Die Einschränkung `numpy<2` stellt sicher, dass NumPy 1.x verwendet wird.

### Installation auf dem Cluster

```bash
# 1. Conda-Umgebung aktivieren
conda activate breathwork-py310

# 2. Cython vorher installieren (NeMo-Voraussetzung)
pip install cython

# 3. indic-numtowords installieren (NeMo-Abhängigkeit)
pip install git+https://github.com/AI4Bharat/indic-numtowords.git

# 4. Neue requirements installieren
pip install -r requirements.txt

# 5. Überprüfen
python -c "import faster_whisper; print('faster-whisper OK')"
python -c "import nemo; print('NeMo OK')"
```

### Erster Modell-Download (einmalig)

Beim ersten Start werden die NeMo-Modelle automatisch heruntergeladen und lokal gecacht. Das passiert **nur einmal** und braucht eine Internetverbindung:

- `vad_multilingual_marblenet` (~50 MB) → `~/.cache/torch/NeMo/`
- `titanet_large` (~80 MB) → `~/.cache/torch/NeMo/`
- `diar_msdd_telephonic` (~200 MB) → `~/.cache/torch/NeMo/`
- `kredor/punctuate-all` (~500 MB) → `~/.cache/huggingface/`
- `large-v3` Whisper-Modell (~3 GB) → `~/.cache/huggingface/`

**Danach: vollständig offline.** ✅

---

## Phase 2: Konfiguration erweitern (`config.py`)

### Was ist das Problem?

Die aktuelle `config.py` kennt nur Whisper-Einstellungen. Wir müssen:
1. Das Whisper-Modell von `small.en` auf `large-v3` ändern
2. Die Sprache von `en` auf `de` ändern
3. Word Timestamps aktivieren
4. Neue Diarization-Parameter hinzufügen

### Was wird geändert?

Die bestehende `config.py` bleibt vollständig erhalten. Wir ändern nur einzelne Werte und fügen einen neuen Abschnitt am Ende hinzu.

**Änderungen an bestehenden Werten:**

```python
# ALT:
WHISPER_MODEL = "small.en"
LANGUAGE = "en"

# NEU:
WHISPER_MODEL = "large-v3"   # Multilingual, beste Genauigkeit für Deutsch
LANGUAGE = "de"              # Deutsch
```

**Neuer Abschnitt am Ende der config.py:**

```python
# ============================================================================
# DIARIZATION SETTINGS (NEU)
# ============================================================================

# Diarization aktivieren oder deaktivieren
# True  = Speaker-Labels werden erkannt und in alle Outputs geschrieben
# False = Pipeline läuft wie bisher ohne Diarization
ENABLE_DIARIZATION = True

# Vocal Isolation (Demucs) aktivieren
# True  = Musik/Hintergrund wird vor Diarization entfernt (langsamer)
# False = Original-Audio wird direkt verwendet (empfohlen für Interviews)
# Für reine Sprach-Interviews (kein Musik-Hintergrund): False empfohlen
ENABLE_STEMMING = False

# Maximale Anzahl Sprecher
# Wird als Obergrenze für die automatische Sprecher-Erkennung verwendet
# Für unsere Interviews: 4 (meistens 2, manchmal 3-4)
MAX_SPEAKERS = 4

# Batch-Größe für faster-whisper Inferenz
# Reduzieren wenn CUDA out of memory Fehler auftreten
# 0 = kein Batching (langsamer aber stabiler)
WHISPER_BATCH_SIZE = 8

# Zahlen als Wörter transkribieren (statt Ziffern)
# True  = "fünfzehn" statt "15" → verbessert Alignment-Genauigkeit
# False = Ziffern bleiben als Ziffern (Standard)
SUPPRESS_NUMERALS = False

# Speaker-Label Format im Output
# Beispiel: "SPEAKER_00", "SPEAKER_01", ...
SPEAKER_LABEL_FORMAT = "SPEAKER_{:02d}"
```

### Warum `large-v3` statt `small.en`?

| Modell | Sprachen | Genauigkeit Deutsch | Größe | Geschwindigkeit |
|--------|---------|--------------------|----|----------------|
| `small.en` | Nur Englisch | ❌ Nicht nutzbar | 466 MB | Sehr schnell |
| `medium.en` | Nur Englisch | ❌ Nicht nutzbar | 1,5 GB | Schnell |
| `large-v3` | 99 Sprachen | ✅ Sehr gut | 3 GB | Langsamer, aber GPU kompensiert |

Die `.en`-Modelle können physisch kein Deutsch verarbeiten. `large-v3` ist das beste multilinguales Modell und auf dem Cluster mit RTX 6000 gut nutzbar.

### Warum `ENABLE_STEMMING = False`?

Demucs (Vocal Isolation) ist dafür gedacht, Sprache aus Musik herauszufiltern – z.B. bei einem Interview mit Hintergrundmusik. Eure Forschungsaufnahmen sind reine Sprach-Aufnahmen ohne Musik. Demucs würde nur Zeit kosten ohne Mehrwert. Es bleibt als Option erhalten (`ENABLE_STEMMING = True`), ist aber standardmäßig deaktiviert.

---

## Phase 3: Transkription umschreiben (`transcribe.py`)

### Was ist das Problem?

Die aktuelle `transcribe.py` nutzt `openai-whisper`. Die API von `faster-whisper` ist anders:
- `openai-whisper` gibt ein Dictionary zurück: `{"text": "...", "segments": [...]}`
- `faster-whisper` gibt einen **Generator** zurück: Segmente werden nacheinander geliefert

Außerdem müssen wir `word_timestamps=True` aktivieren, damit die Diarization später weiß, welches Wort wann gesprochen wurde.

### Was bleibt gleich?

- Die Klasse `WhisperTranscriber` bleibt bestehen
- Die Methoden `transcribe_file()` und `transcribe_files()` bleiben bestehen
- Der Rückgabewert bleibt ein Dictionary mit `text`, `segments`, `language`
- GPU-Auto-Detection bleibt bestehen
- Alle Quality-Metriken (compression_ratio, no_speech_prob, avg_logprob) bleiben

### Was ändert sich?

Der interne Code der Klasse wird auf die faster-whisper API umgeschrieben. Der Rest der Pipeline merkt davon **nichts**, weil der Rückgabewert identisch strukturiert bleibt.

**Neue `transcribe.py`:**

```python
"""
Transkriptions-Modul mit faster-whisper.

Ersetzt openai-whisper durch faster-whisper für:
- Bessere Performance (~3-4x schneller)
- Kompatibilität mit dem Diarization-Modul
- Word-Level Timestamps für Speaker Diarization
"""

from pathlib import Path
from typing import List, Dict, Optional

import torch
from faster_whisper import WhisperModel, BatchedInferencePipeline

import config
import utils

# Compute-Type je nach Hardware
# float16 = GPU (schneller, etwas weniger präzise als float32)
# int8    = CPU (speichersparend)
COMPUTE_TYPES = {"cuda": "float16", "cpu": "int8"}


class WhisperTranscriber:
    """Wrapper-Klasse für faster-whisper Transkription."""

    def __init__(self, model_name: str = None, device: str = None):
        self.model_name = model_name or config.WHISPER_MODEL
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = COMPUTE_TYPES[self.device]
        self.model = None
        self.pipeline = None

    def load_model(self):
        """Lädt das faster-whisper Modell."""
        if self.model is None:
            print(f"Lade Whisper-Modell '{self.model_name}' auf {self.device}...")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            # BatchedInferencePipeline für schnellere Verarbeitung
            if config.WHISPER_BATCH_SIZE > 0:
                self.pipeline = BatchedInferencePipeline(self.model)
            print("✓ Modell geladen")

    def transcribe_file(self, audio_file: Path) -> Optional[Dict]:
        """
        Transkribiert eine einzelne Audio-Datei.

        Rückgabe: Dictionary mit text, segments, language
        (identische Struktur wie bisher, damit merge_outputs.py
        unverändert bleibt)
        """
        if self.model is None:
            self.load_model()

        try:
            # faster-whisper gibt einen Generator zurück – wir sammeln alles
            if self.pipeline and config.WHISPER_BATCH_SIZE > 0:
                segments_gen, info = self.pipeline.transcribe(
                    str(audio_file),
                    language=config.LANGUAGE,
                    batch_size=config.WHISPER_BATCH_SIZE,
                    word_timestamps=True,   # NEU: für Diarization aktiviert
                )
            else:
                segments_gen, info = self.model.transcribe(
                    str(audio_file),
                    language=config.LANGUAGE,
                    task=config.TASK,
                    temperature=config.TEMPERATURE,
                    vad_filter=True,
                    word_timestamps=True,   # NEU: für Diarization aktiviert
                )

            # Generator in Liste umwandeln (faster-whisper ist lazy)
            segments_list = []
            full_text = ""

            for segment in segments_gen:
                full_text += segment.text

                # Wort-Timestamps extrahieren (für Diarization)
                words = []
                if segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        })

                # Segment-Struktur identisch zu openai-whisper halten
                # damit merge_outputs.py unverändert bleibt
                segments_list.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "compression_ratio": segment.compression_ratio,
                    "no_speech_prob": segment.no_speech_prob,
                    "avg_logprob": segment.avg_logprob,
                    "words": words,  # NEU: Wort-Timestamps
                })

            return {
                "text": full_text.strip(),
                "segments": segments_list,
                "language": info.language,
                # NEU: Audio-Waveform für Diarization mitliefern
                # (wird von diarize.py benötigt)
                "_audio_path": str(audio_file),
            }

        except Exception as e:
            print(f"Fehler bei Transkription von {audio_file.name}: {e}")
            return None

    def transcribe_files(
        self,
        audio_files: List[Path],
        output_dir: Path,
    ) -> Dict[str, Dict]:
        """
        Transkribiert mehrere Audio-Dateien.
        Rückgabe: Dictionary {dateiname_ohne_endung: ergebnis_dict}
        (identisch zu bisheriger Implementierung)
        """
        utils.ensure_dir(output_dir)
        self.load_model()

        transcripts = {}
        total = len(audio_files)
        print(f"\nTranskribiere {total} Audio-Dateien...")

        for i, audio_file in enumerate(audio_files, 1):
            print(f"  [{i}/{total}] {audio_file.name}...", end=" ", flush=True)

            result = self.transcribe_file(audio_file)

            if result:
                transcripts[audio_file.stem] = result
                text = result["text"]
                print(f"✓ ({len(text)} Zeichen, {utils.count_words(text)} Wörter)")
            else:
                print("✗ Fehlgeschlagen")

        print(f"\n✓ Erfolgreich transkribiert: {len(transcripts)}/{total}")
        return transcripts
```

### Was ist der `_audio_path` Schlüssel?

Das ist ein interner Schlüssel (mit `_` Präfix als Konvention für "intern"). Er speichert den Pfad zur Audio-Datei im Ergebnis-Dictionary, damit das Diarization-Modul in Phase 4 direkt auf die Audio-Datei zugreifen kann, ohne sie erneut suchen zu müssen.

---

## Phase 4: Neues Diarization-Modul erstellen (`pipeline/diarize.py`)

### Was ist das Problem?

Das Diarization-Projekt ist als **Standalone-Script** gebaut – es liest Argumente von der Kommandozeile und schreibt direkt in Dateien. Wir brauchen es als **wiederverwendbares Modul**, das von `run_pipeline.py` aufgerufen werden kann.

### Was wird gemacht?

Wir extrahieren die Kern-Logik aus `diarize.py` des Diarization-Projekts und kapseln sie in einer sauberen Klasse `DiarizationProcessor`. Diese Klasse:
- Nimmt eine Audio-Datei und ein Transkript-Ergebnis entgegen
- Gibt strukturierte Diarization-Daten zurück (kein Schreiben in Dateien)
- Lässt sich von `run_pipeline.py` aufrufen

**Neue `pipeline/diarize.py`:**

```python
"""
Diarization-Modul für die Breathwork Transcription Pipeline.

Dieses Modul kapselt die Speaker-Diarization-Logik aus dem
whisper-diarization Projekt als wiederverwendbare Klasse.

Verwendete Modelle (alle lokal nach erstem Download):
- MarbleNet:          Voice Activity Detection (VAD)
- TitaNet Large:      Speaker Embeddings (Stimm-Fingerabdrücke)
- diar_msdd_telephonic: Multi-Scale Diarization Decoder
- ctc-forced-aligner: Präzise Wort-Zeitstempel
- deepmultilingualpunctuation: Satzzeichen-Wiederherstellung
"""

import logging
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import faster_whisper

# ctc-forced-aligner für präzise Wort-Timestamps
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)

# Satzzeichen-Wiederherstellung
from deepmultilingualpunctuation import PunctuationModel

import config

# Sprach-Mapping: Whisper-Kürzel → ISO 639-2 (für ctc-forced-aligner)
LANGS_TO_ISO = {
    "de": "ger", "en": "eng", "fr": "fre", "es": "spa",
    "it": "ita", "nl": "dut", "pt": "por", "pl": "pol",
    # ... (vollständige Liste aus helpers.py des Diarization-Projekts)
}

# Sprachen mit Satzzeichen-Unterstützung
PUNCT_MODEL_LANGS = ["en", "fr", "de", "es", "it", "nl", "pt",
                     "bg", "pl", "cs", "sk", "sl"]


class DiarizationProcessor:
    """
    Verarbeitet Speaker Diarization für eine Audio-Datei.

    Workflow:
    1. Audio laden
    2. (Optional) Vocal Isolation via Demucs
    3. Forced Alignment: Wort-Timestamps präzisieren
    4. NeMo MSDD: Sprecher erkennen und Segmente zuordnen
    5. Punctuation Restoration: Satzzeichen wiederherstellen
    6. Realignment: Sprecherwechsel an Satzgrenzen ausrichten
    """

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None
        print(f"DiarizationProcessor initialisiert auf: {self.device}")

    def load_alignment_model(self):
        """Lädt das CTC Forced Alignment Modell (einmalig)."""
        if self.alignment_model is None:
            print("  Lade Alignment-Modell...")
            self.alignment_model, self.alignment_tokenizer = load_alignment_model(
                self.device,
                dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
            print("  ✓ Alignment-Modell geladen")

    def load_punct_model(self):
        """Lädt das Satzzeichen-Modell (einmalig)."""
        if self.punct_model is None:
            print("  Lade Satzzeichen-Modell...")
            self.punct_model = PunctuationModel(model="kredor/punctuate-all")
            print("  ✓ Satzzeichen-Modell geladen")

    def process(
        self,
        audio_file: Path,
        transcript_result: Dict,
    ) -> Optional[List[Dict]]:
        """
        Führt Speaker Diarization für eine Audio-Datei durch.

        Args:
            audio_file:        Pfad zur (vorverarbeiteten) Audio-Datei
            transcript_result: Ergebnis-Dictionary aus transcribe.py
                               (enthält text, segments, language)

        Returns:
            Liste von Satz-Dictionaries mit Speaker-Labels:
            [
                {
                    "speaker": "SPEAKER_00",
                    "start_time_ms": 0,
                    "end_time_ms": 2340,
                    "text": "Und was haben Sie dabei empfunden?"
                },
                ...
            ]
            Oder None bei Fehler.
        """
        try:
            self.load_alignment_model()

            full_transcript = transcript_result["text"]
            language = transcript_result.get("language", config.LANGUAGE)

            # Audio laden
            print(f"  Lade Audio: {audio_file.name}")
            audio_waveform = faster_whisper.decode_audio(str(audio_file))

            # Optional: Vocal Isolation via Demucs
            if config.ENABLE_STEMMING:
                audio_waveform = self._apply_stemming(audio_file, audio_waveform)

            # Forced Alignment: Wort-Timestamps präzisieren
            print("  Forced Alignment...")
            word_timestamps = self._forced_alignment(
                audio_waveform, full_transcript, language
            )

            # NeMo MSDD Diarization: Sprecher erkennen
            print("  Speaker Diarization (NeMo MSDD)...")
            speaker_ts = self._run_diarization(audio_waveform)

            if speaker_ts is None:
                return None

            # Wörter den Sprechern zuordnen
            wsm = self._get_words_speaker_mapping(word_timestamps, speaker_ts)

            # Satzzeichen wiederherstellen (wenn Sprache unterstützt)
            if language in PUNCT_MODEL_LANGS:
                print("  Satzzeichen wiederherstellen...")
                self.load_punct_model()
                wsm = self._restore_punctuation(wsm)

            # Sprecherwechsel an Satzgrenzen ausrichten
            wsm = self._realign_with_punctuation(wsm)

            # Sätze mit Speaker-Labels zusammenbauen
            sentences = self._get_sentences_speaker_mapping(wsm, speaker_ts)

            # In einheitliches Format umwandeln
            result = []
            for sentence in sentences:
                # Speaker-Label formatieren: "Speaker 0" → "SPEAKER_00"
                speaker_raw = sentence["speaker"]  # z.B. "Speaker 0"
                speaker_num = int(speaker_raw.split()[-1])
                speaker_label = config.SPEAKER_LABEL_FORMAT.format(speaker_num)

                result.append({
                    "speaker": speaker_label,
                    "start_time_ms": sentence["start_time"],
                    "end_time_ms": sentence["end_time"],
                    "text": sentence["text"].strip(),
                })

            print(f"  ✓ {len(set(s['speaker'] for s in result))} Sprecher erkannt")
            return result

        except Exception as e:
            print(f"  ✗ Diarization fehlgeschlagen: {e}")
            logging.exception(e)
            return None

    def _forced_alignment(
        self, audio_waveform, full_transcript: str, language: str
    ) -> List[Dict]:
        """Präzisiert Wort-Timestamps via CTC Forced Alignment."""
        emissions, stride = generate_emissions(
            self.alignment_model,
            torch.from_numpy(audio_waveform)
                .to(self.alignment_model.dtype)
                .to(self.alignment_model.device),
            batch_size=config.WHISPER_BATCH_SIZE,
        )

        iso_lang = LANGS_TO_ISO.get(language, "eng")
        tokens_starred, text_starred = preprocess_text(
            full_transcript,
            romanize=True,
            language=iso_lang,
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self.alignment_tokenizer,
        )

        spans = get_spans(tokens_starred, segments, blank_token)
        word_timestamps = postprocess_results(text_starred, spans, stride, scores)
        return word_timestamps

    def _run_diarization(self, audio_waveform) -> Optional[List[Tuple]]:
        """Führt NeMo MSDD Diarization durch."""
        try:
            # Diarization-Modul aus dem whisper-diarization Projekt importieren
            # Dieses Modul muss im Python-Pfad verfügbar sein
            # (entweder installiert oder Pfad hinzugefügt)
            sys.path.insert(0, str(Path(__file__).parent.parent.parent /
                                   "whisper-diarization"))
            from diarization import MSDDDiarizer

            diarizer = MSDDDiarizer(device=self.device)
            speaker_ts = diarizer.diarize(
                torch.from_numpy(audio_waveform).unsqueeze(0)
            )
            del diarizer
            if self.device == "cuda":
                torch.cuda.empty_cache()
            return speaker_ts

        except Exception as e:
            print(f"  ✗ NeMo Diarization fehlgeschlagen: {e}")
            return None

    def _apply_stemming(self, audio_file: Path, audio_waveform):
        """Vocal Isolation via Demucs (optional)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            return_code = os.system(
                f'python -m demucs.separate -n htdemucs --two-stems=vocals '
                f'"{audio_file}" -o "{temp_dir}" --device "{self.device}"'
            )
            if return_code != 0:
                print("  ⚠ Vocal Isolation fehlgeschlagen, nutze Original-Audio")
                return audio_waveform

            vocal_path = (Path(temp_dir) / "htdemucs" /
                          audio_file.stem / "vocals.wav")
            if vocal_path.exists():
                return faster_whisper.decode_audio(str(vocal_path))
            return audio_waveform

    def _get_words_speaker_mapping(
        self, word_timestamps: List[Dict], speaker_ts: List[Tuple]
    ) -> List[Dict]:
        """Ordnet jedem Wort einen Sprecher zu."""
        s, e, sp = speaker_ts[0]
        wrd_pos, turn_idx = 0, 0
        wrd_spk_mapping = []

        for wrd_dict in word_timestamps:
            ws = int(wrd_dict["start"] * 1000)
            we = int(wrd_dict["end"] * 1000)
            wrd = wrd_dict["text"]
            wrd_pos = ws  # Wort-Startzeit als Anker

            while wrd_pos > float(e):
                turn_idx += 1
                turn_idx = min(turn_idx, len(speaker_ts) - 1)
                s, e, sp = speaker_ts[turn_idx]
                if turn_idx == len(speaker_ts) - 1:
                    e = we

            wrd_spk_mapping.append({
                "word": wrd,
                "start_time": ws,
                "end_time": we,
                "speaker": sp,
            })

        return wrd_spk_mapping

    def _restore_punctuation(self, wsm: List[Dict]) -> List[Dict]:
        """Stellt Satzzeichen im Transkript wieder her."""
        words_list = [x["word"] for x in wsm]
        labeled_words = self.punct_model.predict(words_list, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda x: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", x)

        for word_dict, labeled_tuple in zip(wsm, labeled_words):
            word = word_dict["word"]
            if (word and labeled_tuple[1] in ending_puncts and
                    (word[-1] not in model_puncts or is_acronym(word))):
                word += labeled_tuple[1]
                if word.endswith(".."):
                    word = word.rstrip(".")
                word_dict["word"] = word

        return wsm

    def _realign_with_punctuation(
        self, word_speaker_mapping: List[Dict], max_words: int = 50
    ) -> List[Dict]:
        """Richtet Sprecherwechsel an Satzgrenzen aus."""
        # (Logik aus helpers.py des Diarization-Projekts übernommen)
        sentence_ending_puncts = ".?!"
        is_sentence_end = lambda x: (x >= 0 and
            word_speaker_mapping[x]["word"][-1] in sentence_ending_puncts)

        words_list = [d["word"] for d in word_speaker_mapping]
        speaker_list = [d["speaker"] for d in word_speaker_mapping]
        wsp_len = len(word_speaker_mapping)

        k = 0
        while k < wsp_len:
            if (k < wsp_len - 1 and
                    speaker_list[k] != speaker_list[k + 1] and
                    not is_sentence_end(k)):

                # Satzanfang suchen
                left_idx = k
                while (left_idx > 0 and k - left_idx < max_words and
                       speaker_list[left_idx - 1] == speaker_list[left_idx] and
                       not is_sentence_end(left_idx - 1)):
                    left_idx -= 1

                # Satzende suchen
                right_idx = k
                while (right_idx < wsp_len - 1 and
                       right_idx - k < max_words - k + left_idx - 1 and
                       not is_sentence_end(right_idx)):
                    right_idx += 1

                if min(left_idx, right_idx) == -1:
                    k += 1
                    continue

                spk_labels = speaker_list[left_idx:right_idx + 1]
                mod_speaker = max(set(spk_labels), key=spk_labels.count)

                if spk_labels.count(mod_speaker) < len(spk_labels) // 2:
                    k += 1
                    continue

                speaker_list[left_idx:right_idx + 1] = (
                    [mod_speaker] * (right_idx - left_idx + 1)
                )
                k = right_idx

            k += 1

        return [
            {**word_speaker_mapping[i], "speaker": speaker_list[i]}
            for i in range(wsp_len)
        ]

    def _get_sentences_speaker_mapping(
        self, word_speaker_mapping: List[Dict], spk_ts: List[Tuple]
    ) -> List[Dict]:
        """Gruppiert Wörter zu Sätzen mit Speaker-Labels."""
        import nltk
        try:
            sentence_checker = (nltk.tokenize.PunktSentenceTokenizer()
                                 .text_contains_sentbreak)
        except LookupError:
            nltk.download("punkt")
            sentence_checker = (nltk.tokenize.PunktSentenceTokenizer()
                                 .text_contains_sentbreak)

        s, e, spk = spk_ts[0]
        prev_spk = spk
        snts = []
        snt = {"speaker": f"Speaker {spk}", "start_time": s,
               "end_time": e, "text": ""}

        for wrd_dict in word_speaker_mapping:
            wrd, spk = wrd_dict["word"], wrd_dict["speaker"]
            s, e = wrd_dict["start_time"], wrd_dict["end_time"]

            if spk != prev_spk or sentence_checker(snt["text"] + " " + wrd):
                snts.append(snt)
                snt = {"speaker": f"Speaker {spk}", "start_time": s,
                       "end_time": e, "text": ""}
            else:
                snt["end_time"] = e

            snt["text"] += wrd + " "
            prev_spk = spk

        snts.append(snt)
        return snts
```

### Warum eine eigene Klasse statt das Original-Script zu importieren?

Das Original `diarize.py` ist ein Script, das beim Import sofort ausgeführt wird (es liest Kommandozeilen-Argumente, erstellt Verzeichnisse, etc.). Das würde beim Import in unsere Pipeline sofort Fehler verursachen. Durch die Kapselung in eine Klasse haben wir volle Kontrolle darüber, wann und wie die Diarization ausgeführt wird.

---

## Phase 5: Outputs erweitern (`merge_outputs.py`)

### Was ist das Problem?

Die aktuelle `merge_outputs.py` kennt keine Speaker-Labels. Wir müssen alle Output-Formate erweitern, ohne die bestehende Struktur zu brechen.

### Was bleibt gleich?

- Alle bestehenden Funktionen bleiben erhalten
- Quality Flags bleiben erhalten
- Video-Timestamps bleiben erhalten
- Alle 5 Output-Formate bleiben erhalten

### Was wird hinzugefügt?

Jede Output-Funktion bekommt einen optionalen Parameter `diarization_results`. Wenn dieser `None` ist (Diarization deaktiviert oder fehlgeschlagen), verhält sich die Funktion exakt wie bisher.

**Änderungen an den Funktions-Signaturen:**

```python
# ALT:
def generate_individual_transcripts(paired_files, orphaned_files,
                                    transcripts, output_dir):

# NEU (rückwärtskompatibel durch default=None):
def generate_individual_transcripts(paired_files, orphaned_files,
                                    transcripts, output_dir,
                                    diarization_results=None):
```

**Erweitertes TXT-Format (mit Diarization):**

```
[VIDEO TIMESTAMP: 00:02:05.693]
[SPRECHER: SPEAKER_00, SPEAKER_01]

[SPEAKER_00]: Und was haben Sie dabei empfunden?

[SPEAKER_01]: Es war ein warmes Gefühl, fast wie eine Wärme die sich ausbreitet.

[SPEAKER_00]: Können Sie das genauer beschreiben?
```

**Erweitertes JSON-Format (mit Diarization):**

```json
{
  "audio_file": "note1.wav",
  "has_video_timestamp": true,
  "video_timestamp_sec": 125.693,
  "video_timestamp_formatted": "00:02:05.693",
  "transcription": "Und was haben Sie dabei empfunden? Es war ein warmes Gefühl...",
  "diarization_available": true,
  "speakers_detected": ["SPEAKER_00", "SPEAKER_01"],
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
  "quality_flags": [],
  "segments": [...]
}
```

**Hilfsfunktion für Speaker-Text-Formatierung (neu in merge_outputs.py):**

```python
def format_diarization_as_text(diarization_result: List[Dict]) -> str:
    """
    Formatiert Diarization-Ergebnis als lesbaren Text.

    Beispiel-Output:
        [SPEAKER_00]: Und was haben Sie dabei empfunden?

        [SPEAKER_01]: Es war ein warmes Gefühl...
    """
    if not diarization_result:
        return ""

    lines = []
    prev_speaker = None

    for segment in diarization_result:
        speaker = segment["speaker"]
        text = segment["text"].strip()

        if speaker != prev_speaker:
            if lines:
                lines.append("")  # Leerzeile zwischen Sprecherwechseln
            lines.append(f"[{speaker}]: {text}")
        else:
            # Gleicher Sprecher: Text anhängen
            lines[-1] += f" {text}"

        prev_speaker = speaker

    return "\n".join(lines)
```

---

## Phase 6: Pipeline orchestrieren (`run_pipeline.py`)

### Was ist das Problem?

Die aktuelle `run_pipeline.py` kennt keinen Diarization-Schritt. Wir müssen ihn als **optionalen Schritt** zwischen Transkription und Output-Generierung einfügen.

### Was bleibt gleich?

- Alle 9 bestehenden Schritte bleiben erhalten
- Alle Kommandozeilen-Argumente bleiben erhalten
- Single/Multiple Session Erkennung bleibt erhalten

### Was wird hinzugefügt?

Ein neuer **Schritt 3b** zwischen Transkription (Schritt 3) und Output-Generierung (Schritt 4):

```python
# Schritt 3b: Diarization (optional)
diarization_results = {}

if config.ENABLE_DIARIZATION:
    print("\nSchritt 3b: Speaker Diarization...")
    from diarize import DiarizationProcessor

    diarizer = DiarizationProcessor()

    for file_info in paired_files:
        audio_file = file_info['audio']
        filename_stem = audio_file.stem

        if filename_stem not in transcripts:
            continue

        print(f"  Diarisiere: {audio_file.name}")

        # Vorverarbeitete Audio-Datei verwenden
        preprocessed_audio = normalized_dir / audio_file.name

        result = diarizer.process(
            audio_file=preprocessed_audio,
            transcript_result=transcripts[filename_stem],
        )

        if result:
            diarization_results[filename_stem] = result
            speakers = set(s["speaker"] for s in result)
            print(f"  ✓ {len(speakers)} Sprecher: {', '.join(sorted(speakers))}")
        else:
            print(f"  ⚠ Diarization fehlgeschlagen für {audio_file.name}")

    print(f"\n✓ Diarization abgeschlossen: "
          f"{len(diarization_results)}/{len(paired_files)} Dateien")
else:
    print("\nSchritt 3b: Diarization deaktiviert (ENABLE_DIARIZATION=False)")
```

**Neues Kommandozeilen-Argument:**

```python
parser.add_argument(
    '--no-diarization',
    action='store_true',
    help="Diarization deaktivieren (nur Transkription)"
)
```

**Erweiterter Aufruf der Output-Funktionen:**

```python
# Schritt 4: Individuelle Transkripte (mit optionalen Diarization-Daten)
merge_outputs.generate_individual_transcripts(
    paired_files,
    orphaned_files,
    transcripts,
    output_dir,
    diarization_results=diarization_results  # NEU
)
```

---

## Zusammenfassung aller Datei-Änderungen

| Datei | Art der Änderung | Aufwand |
|-------|-----------------|---------|
| `requirements.txt` | Komplett neu schreiben | Gering |
| `config.py` | Werte ändern + neuen Abschnitt hinzufügen | Gering |
| `transcribe.py` | Komplett neu schreiben (gleiche Schnittstelle) | Mittel |
| `pipeline/diarize.py` | Neu erstellen | Hoch |
| `merge_outputs.py` | Funktionen erweitern (rückwärtskompatibel) | Mittel |
| `run_pipeline.py` | Schritt 3b einfügen + Argumente erweitern | Gering |
| `preprocess_audio.py` | Keine Änderung | – |
| `utils.py` | Keine Änderung | – |

---

## Reihenfolge der Umsetzung

```
Phase 1: requirements.txt  →  Testen: pip install läuft durch
    ↓
Phase 2: config.py         →  Testen: python config.py zeigt neue Werte
    ↓
Phase 3: transcribe.py     →  Testen: Einzelne Datei transkribieren, Output prüfen
    ↓
Phase 4: diarize.py        →  Testen: Einzelne Datei diarisieren, Speaker-Labels prüfen
    ↓
Phase 5: merge_outputs.py  →  Testen: Output-Dateien auf Speaker-Labels prüfen
    ↓
Phase 6: run_pipeline.py   →  Testen: Komplette Pipeline auf Test-Session
```

**Wichtig:** Jede Phase einzeln testen, bevor die nächste beginnt. So können Fehler früh lokalisiert werden.

---

## Testen der fertigen Pipeline

### Minimaler Test (lokal, CPU)

```bash
conda activate breathwork-py310

# Einzelne Test-Datei
python pipeline/run_pipeline.py \
    --input /pfad/zu/test_session \
    --no-cleanup

# Output prüfen:
# - transcripts/note_001.txt  → enthält [SPEAKER_00]: und [SPEAKER_01]:
# - transcripts/note_001.json → enthält "diarization": [...]
# - combined_transcript.txt   → alle Sprecher korrekt zugeordnet
```

### Vollständiger Test (Cluster, GPU)

```bash
srun --partition=GPUshortx86 --nodelist=esi-svhpc107 --gpus=1 --pty $SHELL
module load conda
conda activate breathwork-py310

python pipeline/run_pipeline.py \
    --input /pfad/zu/test_session
```

### Diarization deaktivieren (Fallback)

```bash
# Falls Diarization Probleme macht: nur Transkription
python pipeline/run_pipeline.py \
    --input /pfad/zu/session \
    --no-diarization
```

---

## Bekannte Risiken & Lösungen

| Risiko | Wahrscheinlichkeit | Lösung |
|--------|-------------------|--------|
| NeMo-Installation schlägt fehl | Mittel | Cython vorher installieren, numpy<2 beachten |
| CUDA out of memory | Mittel | `WHISPER_BATCH_SIZE` reduzieren, ggf. Demucs deaktivieren |
| Diarization erkennt falsche Sprecher-Anzahl | Gering | `MAX_SPEAKERS` anpassen |
| ctc-forced-aligner Fehler bei Deutsch | Gering | ISO-Code `"ger"` korrekt gesetzt |
| Punctuation-Modell Download schlägt fehl | Gering | Einmalig mit Internet, dann offline |
| Lange Verarbeitungszeit | Hoch (CPU) | GPU auf Cluster verwenden |

---

## Offene Punkte nach Implementierung

- [ ] Modell-Download einmalig auf dem Cluster durchführen (Internet nötig)
- [ ] Test mit echten Forschungsdaten (Deutsch, 2 Sprecher)
- [ ] Qualität der Sprecher-Zuordnung manuell prüfen (Stichprobe)
- [ ] SLURM Job-Script für Batch-Verarbeitung erstellen
- [ ] Dokumentation in README.md aktualisieren

---

*Dieser Plan wird während der Implementierung bei Bedarf angepasst.*
