"""
Transcription module using faster-whisper.

This module handles loading the transcription model and transcribing audio
files while keeping a backward-compatible result structure for downstream
pipeline steps.
"""

from pathlib import Path
from typing import List, Dict, Optional

import torch
from faster_whisper import BatchedInferencePipeline, WhisperModel

import config
import utils


class WhisperTranscriber:
    """Wrapper class for faster-whisper transcription."""

    def __init__(self, model_name: str = None, device: str = None):
        """
        Initialize Whisper transcriber.

        Args:
            model_name: Whisper model to use (defaults to config.WHISPER_MODEL)
            device: "cuda" or "cpu" (if None, auto-detect)
        """
        self.model_name = model_name or config.WHISPER_MODEL
        # Auto-select: GPU if available, else CPU
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        self.model = None
        self.pipeline = None

    def load_model(self):
        """Load the faster-whisper model."""
        if self.model is None:
            print(f"Loading Whisper model '{self.model_name}' on {self.device}...")
            print("(This may take a moment on first run while downloading the model)")
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
            if config.WHISPER_BATCH_SIZE > 0:
                self.pipeline = BatchedInferencePipeline(self.model)
            print("✓ Model loaded successfully")

    def transcribe_file(self, audio_file: Path) -> Optional[Dict]:
        """
        Transcribe a single audio file.

        Args:
            audio_file: Path to audio file

        Returns:
            Result dictionary with text, segments, and language,
            or None if error
        """
        if self.model is None:
            self.load_model()

        try:
            if self.pipeline and config.WHISPER_BATCH_SIZE > 0:
                segments_iter, info = self.pipeline.transcribe(
                    str(audio_file),
                    language=config.LANGUAGE,
                    batch_size=config.WHISPER_BATCH_SIZE,
                    word_timestamps=True,
                )
            else:
                segments_iter, info = self.model.transcribe(
                    str(audio_file),
                    language=config.LANGUAGE,
                    task=config.TASK,
                    temperature=config.TEMPERATURE,
                    word_timestamps=True,
                    vad_filter=True,
                )

            segments_list = []
            full_text = ""

            for segment in segments_iter:
                seg_text = segment.text or ""
                full_text += seg_text

                words = []
                if segment.words:
                    for word in segment.words:
                        words.append({
                            "word": word.word,
                            "start": word.start,
                            "end": word.end,
                            "probability": word.probability,
                        })

                # Keep keys expected by existing quality/output code.
                segments_list.append({
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": seg_text.strip(),
                    "compression_ratio": getattr(segment, "compression_ratio", 1.0),
                    "no_speech_prob": getattr(segment, "no_speech_prob", 0.0),
                    "avg_logprob": getattr(segment, "avg_logprob", -0.5),
                    "words": words,
                })

            return {
                "text": full_text.strip(),
                "segments": segments_list,
                "language": getattr(info, "language", config.LANGUAGE),
                "_audio_path": str(audio_file),
            }

        except Exception as e:
            print(f"Error transcribing {audio_file.name}: {e}")
            return None

    def transcribe_files(
        self,
        audio_files: List[Path],
        output_dir: Path,
    ) -> Dict[str, Dict]:
        """
        Transcribe multiple audio files.

        Args:
            audio_files: List of audio file paths
            output_dir: Directory to save individual transcripts

        Returns:
            Dictionary mapping filename stems to result dictionaries
            (containing text, segments, language, etc.)
        """
        utils.ensure_dir(output_dir)

        # Load model once
        self.load_model()

        transcripts = {}
        total = len(audio_files)

        print(f"\nTranscribing {total} audio files...")

        for i, audio_file in enumerate(audio_files, 1):
            print(f"  [{i}/{total}] {audio_file.name}...", end=" ", flush=True)

            # Transcribe
            result = self.transcribe_file(audio_file)

            if result:
                # Key by filename stem (without extension and path) for easier matching
                transcripts[audio_file.stem] = result

                # Extract text for display
                text = result["text"].strip()
                print(f"✓ ({len(text)} chars, {utils.count_words(text)} words)")
            else:
                print("✗ Failed")

        print(f"\n✓ Successfully transcribed {len(transcripts)}/{total} files")

        return transcripts


if __name__ == "__main__":
    print("Run the main pipeline via 'python pipeline/run_pipeline.py --input <path>'.")
