# Breathwork Transcription Pipeline

**Automated speech-to-text transcription for voice recordings with video timestamps**

🔒 **100% Local Processing** - Your data never leaves your computer
🔬 **Scientifically Valid** - Reproducible, deterministic results
📊 **Complete Metadata** - Timestamps, statistics, and audit trail

---

## Table of Contents

- [What This Does](#what-this-does)
- [How It Works](#how-it-works)
- [Privacy & Data Security](#privacy--data-security)
- [Quick Start](#quick-start)
- [Installation](#installation)
- [Usage](#usage)
- [Input File Requirements](#input-file-requirements)
- [Output Files](#output-files)
- [Technologies Used](#technologies-used)
- [License](#license)
- [Troubleshooting](#troubleshooting)

---

## What This Does

This pipeline automatically transcribes audio recordings (voice notes) and links them to specific timestamps in video recordings. It's designed for:

- **Micro-phenomenological interviews** - Research method where participants describe subjective experiences
- **Video annotation workflows** - Linking voice notes to specific moments in video
- **Qualitative research** - Transcribing interview recordings with timestamps
- **Any timestamped audio** - General-purpose transcription with temporal metadata

**Input:** Audio files (.wav) + JSON files with video timestamps
**Output:** Text transcripts with timestamps, combined transcripts, metadata reports

---

## How It Works

### Processing Details

- **Audio Preprocessing**: FFmpeg converts audio to Whisper-optimized format (16kHz mono, 16-bit PCM) with loudness normalization
- **Transcription**: Whisper AI model processes audio locally, no internet required
- **Quality Detection**: Automatic hallucination detection, silence detection, and confidence scoring
- **Verbatim Output**: Preserves filler words, hesitations, and natural speech patterns
- **Deterministic**: Temperature 0.0 ensures identical results on repeated runs

---

## Privacy & Data Security

### Data Stays Local


#### 1. No Network Dependencies in the Pipeline

- All pipeline scripts operate offline and do not initiate network or HTTP requests.

- No external API calls, telemetry, or analytics are included in any part of the workflow.
#### 4. **Verify Yourself**


**Perfect for:**
- HIPAA-regulated healthcare research
- GDPR-compliant European research
- IRB-approved sensitive research
- Confidential interview data
- Personal recordings

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/Namsjain01/breathwork-transcription.git
cd breathwork-transcription

# 2. Create virtual environment
python3 -m venv .venv

# 3. Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR .venv\Scripts\activate for Windows

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run pipeline on your recordings
python3 pipeline/run_pipeline.py --input /path/to/your/recordings

# Output will be in: /path/to/your/recordings/transcripts/
```

**Speaker Diarization (optional):** To get speaker labels (`[SPEAKER_00]:`, `[SPEAKER_01]:`, …), clone `whisper-diarization` as a **sibling** of this repo, then install and run as usual:

```bash
cd /path/to/parent/dir
git clone https://github.com/Namsjain01/breathwork-transcription.git
git clone https://github.com/MahmoudAshraf97/whisper-diarization.git
cd breathwork-transcription
pip install -r requirements.txt
python3 pipeline/run_pipeline.py --input /path/to/your/recordings
```

Without the sibling repo, run with `--no-diarization` for transcription only. See [Cluster setup](README_cluster_for_new_users.md) for cluster-specific steps.

---

## Installation

### Prerequisites

- **Python 3.8 or higher**
- **FFmpeg** (audio processing)
- **8GB+ RAM recommended** (for Whisper model)

### Setup

```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate  # macOS/Linux
# OR .venv\Scripts\activate for Windows

# Install dependencies
pip install -r requirements.txt

# Install ffmpeg

brew install ffmpeg # macOS/Linux

# for Linux
sudo apt update
sudo apt install ffmpeg

# for Windows
 Download from https://ffmpeg.org/download.html and add to PATH

# Verify configuration
python3 pipeline/config.py
```

---

## Usage

### Basic Usage

Point the pipeline at ANY folder containing .wav and .json files:

```bash
python3 pipeline/run_pipeline.py --input /path/to/your/recordings
```

Transcripts will be created in `/path/to/your/recordings/transcripts/`

### Single Session

Process one folder with recordings:

```bash
# Your folder structure:
my_recording_session/
├── note1.wav
├── note1.json
├── note2.wav
├── note2.json
└── note3.wav
    note3.json

# Run pipeline
python3 pipeline/run_pipeline.py --input /path/to/my_recording_session

# Output created at:
my_recording_session/transcripts/
├── transcripts/              # Individual transcript files
│   ├── note1.txt
│   ├── note1.json
│   ├── note2.txt
│   └── ...
├── combined_transcript.txt   # All transcripts together
├── combined_transcript.json  # Machine-readable format
├── plain_text_transcript.txt # Plain text only (no timestamps/metadata)
└── processing_report.txt     # Statistics and metadata
```

### Multiple Sessions (Batch Processing)

Process multiple session folders at once:

```bash
# Your folder structure:
all_my_sessions/
├── session_2024_01_15/
│   ├── note1.wav
│   └── note1.json
├── session_2024_01_20/
│   ├── note1.wav
│   └── note1.json
└── session_2024_02_01/
    ├── note1.wav
    └── note1.json

# Run pipeline
python3 pipeline/run_pipeline.py --input /path/to/all_my_sessions

# Output created in each session folder:
session_2024_01_15/transcripts/
session_2024_01_20/transcripts/
session_2024_02_01/transcripts/
```

### Advanced Options

```bash
# Process specific session by name
python3 pipeline/run_pipeline.py --input /path/to/all_sessions --session session_2024_01_15

# Keep intermediate audio files (for debugging)
python3 pipeline/run_pipeline.py --input /path/to/folder --no-cleanup

# Use current directory
cd /path/to/my/recordings
python3 /path/to/pipeline/run_pipeline.py --input .

# Get help
python3 pipeline/run_pipeline.py --help
```

---

## Input File Requirements

### Required File Structure

Each audio recording needs a matching JSON file:

```
note_001.wav  ←→  note_001.json
```

### JSON Format

The JSON file must contain a video timestamp in seconds:

```json
{
  "video_timestamp_sec": 125.693
}
```

This number represents the timestamp in the video where this voice note was recorded.



### Audio Format

- **Supported formats**: `.wav`, `.WAV`
- **Any sample rate** (pipeline converts to 16kHz)
- **Any channels** (pipeline converts to mono)
- **Any bit depth** (pipeline converts to 16-bit)

### Files Without Timestamps

Audio files WITHOUT matching JSON files will still be transcribed but marked as "orphaned" (no video timestamp).

---

## Output Files


### Individual Transcript (TXT)

**note1.txt:**
```
[VIDEO TIMESTAMP: 00:02:05.693]

This is the transcribed text from the audio recording.
All words are preserved verbatim including um, uh, etc.
```

**With quality warnings (if detected):**
```
[VIDEO TIMESTAMP: 00:02:05.693]
[QUALITY WARNINGS: hallucination_detected]

Thank you for watching. Please like and subscribe.
```

### Individual Transcript (JSON)

**note1.json:**
```json
{
  "audio_file": "note1.wav",
  "has_video_timestamp": true,
  "video_timestamp_sec": 125.693,
  "video_timestamp_formatted": "00:02:05.693",
  "audio_duration_sec": 7.234,
  "language": "en",
  "transcription": "This is the transcribed text...",
  "word_count": 42,
  "character_count": 215,
  "quality_flags": [],
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 7.234,
      "text": "This is the transcribed text...",
      "compression_ratio": 1.13,
      "no_speech_prob": 0.001,
      "avg_logprob": -0.287,
      "confidence": 0.751,
      "likely_hallucination": false,
      "likely_silence": false,
      "low_confidence": false,
      "quality_flags": []
    }
  ]
}
```

### Combined Transcript (TXT)

All transcripts merged into one file, sorted by timestamp:

```
================================================================================
MICRO-PHENOMENOLOGICAL INTERVIEW TRANSCRIPT
================================================================================
Session: my_recording_session
Total Recordings: 5
Transcription Model: Whisper small.en
Processing Date: 2025-11-11 10:30:15
================================================================================

────────────────────────────────────────────────────────────────────────────────
ANNOTATION #1
VIDEO TIMESTAMP: 00:00:05.693 (5.693 seconds)
AUDIO FILE: note1.wav
DURATION: 6.2 seconds
────────────────────────────────────────────────────────────────────────────────

First transcribed text here...

────────────────────────────────────────────────────────────────────────────────
ANNOTATION #2
VIDEO TIMESTAMP: 00:02:15.420 (135.420 seconds)
...
```

### Combined Transcript (JSON)

Complete structured data:

```json
{
  "session_metadata": {
    "session_id": "my_recording_session",
    "processing_timestamp": "2025-11-11T10:30:15",
    "transcription_model": "whisper-small.en",
    "total_recordings": 5,
    "pipeline_version": "1.0.0",
    "quality_checks_enabled": true
  },
  "annotations": [
    {
      "id": 1,
      "video_timestamp_sec": 5.693,
      "audio_file": "note1.wav",
      "transcription": "...",
      "word_count": 42,
      "quality_flags": [],
      "segments": [...]
    },
    ...
  ],
  "statistics": {
    "total_audio_duration_sec": 187.5,
    "total_words": 324,
    "average_words_per_annotation": 65,
    "quality_metrics": {
      "total_segments": 15,
      "segments_with_hallucination": 0,
      "segments_with_silence": 1,
      "segments_with_low_confidence": 0,
      "recordings_with_hallucination": 0,
      "recordings_with_silence": 1,
      "recordings_with_low_confidence": 0
    }
  }
}
```

### Plain Text Transcript

**plain_text_transcript.txt:**

All transcribed text concatenated together without timestamps or metadata. Perfect for copy/paste or text analysis.

```
This is the first transcription text.

This is the second transcription text.
```

### Processing Report

**processing_report.txt:**
```
================================================================================
TRANSCRIPTION PROCESSING REPORT
================================================================================
Session: my_recording_session
Date: 2025-11-11 10:30:15
Model: Whisper small.en (244M parameters)
================================================================================

FILES PROCESSED
────────────────────────────────────────────────────────────────────────────────
Total audio files found:        5
Successfully transcribed:       5
Success rate:                   100.0%

TIMING
────────────────────────────────────────────────────────────────────────────────
Total processing time:          2m 15s
Average per file:               27.0s

CONTENT STATISTICS
────────────────────────────────────────────────────────────────────────────────
Total audio duration:           187.5 seconds
Total words transcribed:        324
Average words per recording:    65

QUALITY METRICS
────────────────────────────────────────────────────────────────────────────────
Total segments analyzed:        15
Segments with hallucination:    0 (0.0%)
Segments with silence:          1 (6.7%)
Segments with low confidence:   0 (0.0%)
Recordings with quality issues: 1 (20.0%)

QUALITY THRESHOLDS
────────────────────────────────────────────────────────────────────────────────
Compression ratio threshold:    2.4 (hallucination detector)
No-speech probability threshold: 0.6 (silence detector)
Confidence threshold:           -1.0 (low confidence)
```

---

## Quality Detection & Configuration

### What Gets Detected

The pipeline automatically analyzes every transcription segment for quality issues:

#### 1. **Hallucination Detection**
Detects when Whisper generates fake or repetitive text (common on silence or noise).

**Examples:**
- ✗ "Thank you for watching. Please like and subscribe." (on silence)
- ✗ "The same thing. The same thing. The same thing." (repetition loop)

**How it works:** Compression ratio > 2.4 indicates hallucination

#### 2. **Silence Detection**
Identifies segments with no actual speech (breathing, background noise, pauses).

**How it works:** no_speech_prob > 0.6 indicates likely silence

#### 3. **Low Confidence Detection**
Flags transcriptions where Whisper was uncertain about the words.

**How it works:** avg_logprob < -1.0 indicates low confidence (< 37%)

### Configuring Quality Thresholds

Edit `pipeline/config.py` to adjust detection sensitivity:

```python
# Enable/disable quality checks
ENABLE_QUALITY_CHECKS = True  # Set to False to disable

# Hallucination detection (higher = more lenient)
COMPRESSION_RATIO_THRESHOLD = 2.4  # Default: 2.4
# Examples:
#   2.0 = Strict (catches more, may have false positives)
#   2.4 = Balanced (recommended)
#   3.0 = Lenient (only obvious hallucinations)

# Silence detection (higher = more lenient)
NO_SPEECH_THRESHOLD = 0.6  # Default: 0.6
# Examples:
#   0.4 = Strict (flags more segments as silence)
#   0.6 = Balanced (recommended)
#   0.8 = Lenient (keeps quiet speech)

# Confidence threshold (lower = more lenient)
CONFIDENCE_THRESHOLD = -1.0  # Default: -1.0 (37% confidence)
# Examples:
#   -0.5 = Strict (60% minimum confidence)
#   -1.0 = Balanced (37% minimum confidence)
#   -1.5 = Lenient (22% minimum confidence)
```

### Understanding Quality Flags

When quality issues are detected, you'll see flags in the output:

- **`hallucination_detected`** - Likely fake/repetitive text
- **`silence_detected`** - Likely no actual speech
- **`low_confidence`** - Uncertain transcription

**In TXT files:**
```
[QUALITY WARNINGS: hallucination_detected]
```

**In JSON files:**
```json
{
  "quality_flags": ["hallucination_detected"],
  "segments": [
    {
      "compression_ratio": 3.5,
      "likely_hallucination": true
    }
  ]
}
```

### Quality Metrics in Reports

Processing reports now include quality statistics:
- Total segments analyzed
- Percentage of segments with each issue type
- Number of recordings affected
- Thresholds used for detection

---

## Technologies Used

### Core Technologies

| Component | Technology | Purpose | License |
|-----------|-----------|---------|---------|
| **Transcription AI** | OpenAI Whisper (small.en model) | Speech-to-text recognition | MIT License |
| **Audio Processing** | FFmpeg | Audio format conversion & normalization | LGPL/GPL |
| **Python Runtime** | Python 3.8+ | Pipeline execution | PSF License |
| **ML Framework** | PyTorch | Neural network inference (for Whisper) | BSD License |

### Whisper Model Details

- **Model**: `small.en` (English-only)
- **Parameters**: 244 million
- **Size**: 466 MB
- **Accuracy**: High-quality transcription for English
- **Speed**: ~3-5x realtime on CPU, faster on GPU
- **License**: MIT (fully open source)

### Why These Technologies?

1. **Whisper** - State-of-the-art transcription accuracy, runs locally, open source
2. **FFmpeg** - Industry-standard audio processing, handles all audio formats
3. **Python** - Cross-platform, extensive scientific computing ecosystem
4. **PyTorch** - Efficient neural network inference, GPU acceleration support

---

## License

### This Project

**MIT License** - You can use, modify, and distribute this code freely.

See [LICENSE](LICENSE) file for full text.

### Dependencies

All dependencies are open source and permissively licensed:

- **Whisper**: MIT License (OpenAI)
- **PyTorch**: BSD License (Meta/Facebook)
- **FFmpeg**: LGPL/GPL (depends on build configuration)
- **NumPy**: BSD License

**No proprietary software required.**

---

## Troubleshooting

### Poor Transcription Quality (Missing or Wrong Words)

**Issue:** Transcriptions are inaccurate, missing words, or contain wrong words

**Cause:** The current Whisper model (small.en) may not be accurate enough for your audio quality or speech characteristics.

**Solution 1: Use a Larger Model (Recommended)**

Edit `pipeline/config.py` and change the model:

```python
# For better accuracy (recommended)
WHISPER_MODEL = "medium.en"  # 769M params, ~3x slower but much more accurate

# For maximum accuracy (if you have time and resources)
WHISPER_MODEL = "large-v3"   # 1.5B params, ~10x slower but best accuracy
```

**Model Comparison:**

| Model | Size | RAM | Accuracy | Speed | Best For |
|-------|------|-----|----------|-------|----------|
| small.en | 466MB | 2GB | Good | Fast | Clear speech, good audio |
| medium.en | 1.5GB | 5GB | Better | 3x slower | Most use cases (recommended) |
| large-v3 | 2.9GB | 10GB | Best | 10x slower | Difficult audio, accents, technical terms |

**After changing the model, run the pipeline again:**
```bash
python3 pipeline/run_pipeline.py --input /path/to/your/recordings
```

The new model will be downloaded automatically on first use.

**Solution 2: Improve Audio Quality**

- Ensure good microphone placement
- Reduce background noise
- Speak clearly and at moderate pace
- Check audio levels (not too quiet or distorted)

**Solution 3: Check Quality Flags**

Review the processing report for quality warnings:
- Look for `hallucination_detected` flags
- Check `low_confidence` segments
- Review segments marked as `silence_detected`

These flags in `transcripts/processing_report.txt` can help identify problematic sections.

### FFmpeg Not Found

**Error:**
```
EnvironmentError: FFmpeg not found
```

**Solution:**
```bash
# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg

# Verify installation
ffmpeg -version
```

### Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:** Whisper model requires ~2GB RAM. Close other applications or use a smaller model:

Edit `pipeline/config.py`:
```python
WHISPER_MODEL = "tiny.en"  # Smaller model (39MB, less accurate)
# or
WHISPER_MODEL = "base.en"  # Medium model (74MB)
```

### Slow Processing

**Issue:** Transcription taking too long

**Solutions:**
1. **Use GPU**: If you have NVIDIA GPU, install PyTorch with CUDA support (10-50x faster)
2. **Smaller model**: Use `tiny.en` or `base.en` (faster but less accurate)
3. **Parallel preprocessing**: Already enabled by default

### No Audio Files Found

**Error:**
```
FileNotFoundError: No audio files found
```

**Solution:**
- Ensure your folder contains `.wav` or `.WAV` files
- Check file extensions are correct
- Verify you're pointing to the right folder

### Python Version Issues

**Error:**
```
SyntaxError or ModuleNotFoundError
```

**Solution:** Requires Python 3.8 or higher:
```bash
python3 --version  # Should show 3.8 or higher
```

### Still Having Issues?

1. Check the processing report in `transcripts/processing_report.txt`
2. Run with `--no-cleanup` to inspect intermediate files
3. Test with a single small audio file first
4. Verify `python3 pipeline/config.py` runs without errors

---

## Additional Information

### System Requirements

- **OS**: macOS, Linux, Windows
- **Python**: 3.8 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 2GB for model + space for your audio files
- **CPU**: Any modern CPU (faster is better)
- **GPU**: Optional but significantly faster with CUDA-capable NVIDIA GPU

### Performance Expectations

- **Preprocessing**: ~1-2 seconds per audio file
- **Transcription**: ~3-5x realtime on CPU (10 seconds of audio = 30-50 seconds processing)
- **GPU**: 10-50x realtime (much faster if available)

### Scientific Use

This pipeline is designed for research:

✅ **Reproducible**: Temperature 0.0 ensures identical results on repeated runs
✅ **Verbatim**: Preserves natural speech including filler words
✅ **Auditable**: Complete processing reports with timestamps and statistics
✅ **Open Source**: All code and models are open source and inspectable
✅ **No Modification**: Transcriptions are not edited or "cleaned up"

Perfect for:
- Qualitative research
- Phenomenological studies
- Interview analysis
- Conversation analysis
- Linguistic research



---

**Questions or Issues?** Open an issue on GitHub: https://github.com/Namsjain01/breathwork-transcription/issues
