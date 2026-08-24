# Implementierungsstand: Speaker Diarization

> **Stand:** 2026-03-10  
> **Referenz:** [doc/implementierungsplan.md](implementierungsplan.md) (6 Phasen)

---

## Kurzfassung

**Phasen 1–5 sind umgesetzt.**  
**Phase 6 ist umgesetzt (2026-03-10).** Die Diarization-Integration ist abgeschlossen.

---

## Detail-Check pro Phase

| Phase | Beschreibung              | Status | Anmerkung |
|-------|---------------------------|--------|-----------|
| 1     | `requirements.txt`        | ✅     | faster-whisper, nemo, nltk, demucs, ctc-forced-aligner, deepmultilingualpunctuation, indic-numtowords, numpy<2 |
| 2     | `config.py`               | ✅     | ENABLE_DIARIZATION, ENABLE_STEMMING, MAX_SPEAKERS, SPEAKER_LABEL_FORMAT, WHISPER_BATCH_SIZE, large-v3 |
| 3     | `transcribe.py`           | ✅     | faster_whisper, word_timestamps=True, _audio_path im Ergebnis |
| 4     | `pipeline/diarize.py`     | ✅     | DiarizationProcessor, process(), Forced Alignment, MSDD, Punctuation, Realignment |
| 5     | `merge_outputs.py`        | ✅     | diarization_results in allen Output-Funktionen, format_diarization_as_text, get_diarization_for_file |
| 6     | `run_pipeline.py`         | ✅     | Schritt 3b, diarization_results an alle merge_outputs, --no-diarization |

---

## Phase 6 umgesetzt (Inhalt)

- **Schritt 3b:** Nach der Transkription wird bei `config.ENABLE_DIARIZATION` und ohne `--no-diarization` der `DiarizationProcessor` ausgeführt (für alle Dateien mit Transkript, inkl. Orphans). Verwendet wird die vorverarbeitete Audio in `normalized_dir`.
- **`--no-diarization`:** Deaktiviert Diarization für den aktuellen Lauf.
- **`diarization_results`** wird an `generate_individual_transcripts`, `generate_combined_txt`, `generate_combined_json` und `generate_plain_text` übergeben.

---

## Nächste Schritte / Tests

- End-to-End-Test mit einer Session (z. B. `python pipeline/run_pipeline.py --input /pfad/zur/session --no-cleanup`).
- Prüfen: In `transcripts/*.txt` und `combined_transcript.txt`/`.json` erscheinen `[SPEAKER_00]:` etc., sofern Diarization aktiv und erfolgreich.
- Optional: Offene Punkte aus dem Implementierungsplan abarbeiten (Modell-Download auf Cluster, Tests mit echten Daten, README aktualisieren).
