"""
Configuration file for micro-phenomenological interview transcription pipeline.

This file contains all settings and paths used by the transcription system.
Modify these settings to customize the pipeline behavior.
"""

import os
from pathlib import Path

# ============================================================================
# DIRECTORY PATHS
# ============================================================================

# Base directory (project root) - used for models storage only
BASE_DIR = Path(__file__).parent.parent.absolute()

# Models: Whisper model storage (global location)
MODELS_DIR = BASE_DIR / "models"

# Temporary: Processed audio files directory name (created per session)
# This will be created inside each session folder during processing
PROCESSED_DIR_NAME = "processed_audio"

# Output: Transcripts directory name (created per session)
# This will be created inside each session folder for outputs
OUTPUT_DIR_NAME = "transcripts"

# ============================================================================
# WHISPER SETTINGS
# ============================================================================

# Model size: "tiny.en", "base.en", "small.en", "medium.en", "large-v3"
# Accuracy increases with model size, but so does processing time:
#   small.en  = 244M params, 466 MB  - Fast, good accuracy (RECOMMENDED)
#   medium.en = 769M params, 1.5 GB  - Slower, better accuracy
#   large-v3  = 1.5B params, 2.9 GB  - Slowest, best accuracy
WHISPER_MODEL = "large-v3"

# Language (force English transcription)
LANGUAGE = "en"

# Temperature for sampling (0.0 = deterministic/reproducible)
TEMPERATURE = 0.0

# Task type (transcribe vs translate)
TASK = "transcribe"

# Verbose output from Whisper
VERBOSE = False

# Batch size for faster-whisper batched inference
# Reduce this value if you encounter out-of-memory errors.
# Set to 0 to disable batching.
WHISPER_BATCH_SIZE = 8

# Suppress numeral tokens during transcription (used by diarization alignment flow)
SUPPRESS_NUMERALS = False

# ============================================================================
# QUALITY DETECTION SETTINGS
# ============================================================================

# Enable quality checks (hallucination detection, VAD)
ENABLE_QUALITY_CHECKS = True

# Compression ratio threshold for hallucination detection
# Values > 2.4 indicate likely hallucinations or repetitive text
COMPRESSION_RATIO_THRESHOLD = 2.4

# No-speech probability threshold for silence detection
# Values > 0.6 indicate likely silence (no speech detected)
NO_SPEECH_THRESHOLD = 0.6

# Confidence score threshold (based on avg_logprob)
# Values < -1.0 indicate low confidence transcription
CONFIDENCE_THRESHOLD = -1.0

# ============================================================================
# AUDIO PREPROCESSING SETTINGS
# ============================================================================

# Target sample rate (Whisper optimized for 16kHz)
TARGET_SAMPLE_RATE = 16000

# Target channels (1 = mono, Whisper works best with mono)
TARGET_CHANNELS = 1

# Target bit depth (16-bit PCM standard for Whisper)
TARGET_BIT_DEPTH = 16

# Audio normalization (loudness normalization for consistent volume)
NORMALIZE_AUDIO = True

# FFmpeg executable (usually just "ffmpeg" if on PATH)
FFMPEG_EXECUTABLE = "ffmpeg"

# ============================================================================
# FILE MATCHING SETTINGS
# ============================================================================

# Audio file extensions to process
AUDIO_EXTENSIONS = [".wav", ".WAV"]

# JSON metadata file extension
JSON_EXTENSION = ".json"

# Files to ignore (only system files - NOT audio files)
# Note: Only ignore actual system files, not audio recordings
# Even if a recording is named "temp_audio.wav", it should be transcribed
IGNORE_FILES = [".DS_Store", "Thumbs.db", "desktop.ini"]

# ============================================================================
# OUTPUT SETTINGS
# ============================================================================

# Generate individual transcript files
GENERATE_INDIVIDUAL_TRANSCRIPTS = True

# Generate combined transcript (TXT format)
GENERATE_COMBINED_TXT = True

# Generate combined transcript (JSON format)
GENERATE_COMBINED_JSON = True

# Generate processing report
GENERATE_PROCESSING_REPORT = True

# Generate plain text transcript (all text, no timestamps/metadata)
GENERATE_PLAIN_TEXT = True

# Delete intermediate processed audio files after transcription
DELETE_INTERMEDIATE_FILES = True

# ============================================================================
# TRANSCRIPT FORMATTING
# ============================================================================

# Timestamp format for video timestamps
TIMESTAMP_FORMAT = "%H:%M:%S"  # e.g., "00:00:05.693"

# Include milliseconds in timestamp display
INCLUDE_MILLISECONDS = True

# Separator for combined transcript sections
SECTION_SEPARATOR = "─" * 80

# Header separator for combined transcript
HEADER_SEPARATOR = "=" * 80

# ============================================================================
# PROCESSING SETTINGS
# ============================================================================

# Number of parallel processes for audio preprocessing
# Set to None to use all available CPU cores
NUM_PARALLEL_PROCESSES = None

# Skip already transcribed files (for resuming interrupted processing)
SKIP_EXISTING = True

# Show progress bars
SHOW_PROGRESS = True

# ============================================================================
# LOGGING SETTINGS
# ============================================================================

# Log level: "DEBUG", "INFO", "WARNING", "ERROR"
LOG_LEVEL = "INFO"

# Log to file in addition to console
LOG_TO_FILE = True

# Log file location (will be created in session output directory)
LOG_FILE_NAME = "transcription.log"

# ============================================================================
# SCIENTIFIC METADATA
# ============================================================================

# Methodology reference
METHODOLOGY_FILE = BASE_DIR / "METHODOLOGY.md"

# Version of this pipeline
PIPELINE_VERSION = "1.1.0"

# Processing description
PROCESSING_DESCRIPTION = """
Audio files were preprocessed using FFmpeg (loudness normalization,
conversion to 16kHz mono 16-bit PCM) and transcribed using faster-whisper
large-v3 with language='en' and temperature 0.0 for
reproducibility. All processing performed locally with no cloud services.
"""

# ============================================================================
# DIARIZATION SETTINGS
# ============================================================================

# Optional speaker diarization feature toggle
ENABLE_DIARIZATION = True

# Optional Demucs vocal isolation before diarization
# For speech-only interviews this is usually not needed.
ENABLE_STEMMING = False

# Upper bound for automatic speaker estimation
MAX_SPEAKERS = 4

# Speaker label format used in output files
SPEAKER_LABEL_FORMAT = "SPEAKER_{:02d}"

# Path to whisper-diarization repository (required when ENABLE_DIARIZATION is True).
# None = expect whisper-diarization as sibling of this repo:
#   <parent_dir>/breathwork-transcription/
#   <parent_dir>/whisper-diarization/
# Set to a Path or string to use a custom location (e.g. on cluster).
DIARIZATION_REPO_PATH = None

# ============================================================================
# VALIDATION
# ============================================================================

def validate_config():
    """Validate configuration settings and create necessary directories."""

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Validate Whisper model name
    valid_models = ["tiny.en", "tiny", "base.en", "base", "small.en",
                    "small", "medium.en", "medium", "large", "large-v2", "large-v3"]
    if WHISPER_MODEL not in valid_models:
        raise ValueError(f"Invalid Whisper model: {WHISPER_MODEL}. "
                        f"Valid options: {', '.join(valid_models)}")

    # Check if FFmpeg is available
    import subprocess
    try:
        subprocess.run([FFMPEG_EXECUTABLE, "-version"],
                      capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise EnvironmentError(f"FFmpeg not found. Please install FFmpeg and "
                             f"ensure it's on your PATH.")

    print("✓ Configuration validated successfully")
    return True

if __name__ == "__main__":
    # Test configuration when run directly
    validate_config()
    print(f"\nConfiguration Summary:")
    print(f"  Models directory: {MODELS_DIR}")
    print(f"  Whisper model: {WHISPER_MODEL}")
    print(f"  Sample rate: {TARGET_SAMPLE_RATE} Hz")
    print(f"  Channels: {TARGET_CHANNELS} (mono)")
    print(f"  Bit depth: {TARGET_BIT_DEPTH}-bit")
    diar_path = DIARIZATION_REPO_PATH if DIARIZATION_REPO_PATH is not None else (BASE_DIR.parent / "whisper-diarization")
    print(f"  Diarization repo (when enabled): {diar_path}")
    print(f"\nNote: Input/output directories are now specified via --input argument")
