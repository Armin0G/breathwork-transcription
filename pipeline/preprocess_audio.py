"""
Audio preprocessing module for the transcription pipeline.

This module handles audio normalization and format conversion using FFmpeg.
Converts audio to optimal format for Whisper transcription (16kHz mono 16-bit PCM).
"""

import subprocess
from pathlib import Path
from typing import List
import multiprocessing as mp

import config
import utils


def preprocess_single_audio(input_file: Path, output_file: Path) -> bool:
    """
    Preprocess a single audio file using FFmpeg.

    Args:
        input_file: Path to input audio file
        output_file: Path to output processed file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Ensure output directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg command
        cmd = [
            config.FFMPEG_EXECUTABLE,
            '-i', str(input_file),
            '-vn',  # No video
            '-ar', str(config.TARGET_SAMPLE_RATE),  # Sample rate
            '-ac', str(config.TARGET_CHANNELS),  # Channels (mono)
            '-sample_fmt', 's16',  # 16-bit PCM
        ]

        # Add loudness normalization if enabled
        if config.NORMALIZE_AUDIO:
            cmd.extend(['-af', 'loudnorm'])

        # Output file (overwrite if exists)
        cmd.extend(['-y', str(output_file)])

        # Run FFmpeg
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        return True

    except subprocess.CalledProcessError as e:
        print(f"Error preprocessing {input_file.name}: {e.stderr}")
        return False
    except Exception as e:
        print(f"Unexpected error preprocessing {input_file.name}: {e}")
        return False


def preprocess_audio_files(audio_files: List[Path], output_dir: Path,
                           parallel: bool = True) -> List[Path]:
    """
    Preprocess multiple audio files.

    Args:
        audio_files: List of input audio file paths
        output_dir: Directory for processed audio files
        parallel: Whether to process files in parallel

    Returns:
        List of successfully processed output file paths
    """
    utils.ensure_dir(output_dir)

    # Prepare input/output pairs
    pairs = []
    for audio_file in audio_files:
        output_file = output_dir / audio_file.name
        pairs.append((audio_file, output_file))

    # Process files
    if parallel and config.NUM_PARALLEL_PROCESSES != 1:
        # Parallel processing
        num_processes = config.NUM_PARALLEL_PROCESSES or mp.cpu_count()
        num_processes = min(num_processes, len(pairs))

        print(f"Preprocessing {len(pairs)} audio files using {num_processes} processes...")

        with mp.Pool(processes=num_processes) as pool:
            results = pool.starmap(preprocess_single_audio, pairs)
    else:
        # Sequential processing
        print(f"Preprocessing {len(pairs)} audio files sequentially...")
        results = [preprocess_single_audio(inp, out) for inp, out in pairs]

    # Return successful outputs
    successful_outputs = [
        output for (_, output), success in zip(pairs, results) if success
    ]

    print(f"✓ Successfully preprocessed {len(successful_outputs)}/{len(pairs)} files")

    return successful_outputs


if __name__ == "__main__":
    # Test preprocessing
    print("Testing audio preprocessing...")

    # Find test session
    sessions = utils.find_all_sessions(config.INPUT_DIR)
    if not sessions:
        print("No sessions found!")
        exit(1)

    session = sessions[0]
    print(f"\nTesting with session: {session['name']}")

    # Find audio files
    paired, orphaned = utils.find_audio_json_pairs(session['path'])
    if not paired:
        print("No audio files found!")
        exit(1)

    # Test on first file only
    test_file = paired[0]['audio']
    print(f"\nTest preprocessing: {test_file.name}")

    output_dir = config.PROCESSED_DIR / session['name'] / "normalized_audio"
    output_file = output_dir / test_file.name

    success = preprocess_single_audio(test_file, output_file)

    if success:
        print(f"✓ Preprocessing successful!")
        print(f"  Input: {test_file}")
        print(f"  Output: {output_file}")
        print(f"  Output size: {utils.format_file_size(output_file.stat().st_size)}")
    else:
        print("✗ Preprocessing failed!")
