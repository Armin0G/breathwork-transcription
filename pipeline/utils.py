"""
Utility functions for the transcription pipeline.

This module provides helper functions for file matching, timestamp formatting,
and other common operations used throughout the pipeline.
"""

import json
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import config


def find_audio_json_pairs(session_dir: Path) -> Tuple[List[Dict], List[Path]]:
    """
    Find all audio files and match them with JSON timestamp files.

    Args:
        session_dir: Path to session directory containing audio and JSON files

    Returns:
        Tuple of (paired_files, orphaned_files)
        - paired_files: List of dicts with 'audio', 'json', 'timestamp' keys
        - orphaned_files: List of audio file paths without matching JSON
    """
    paired_files = []
    orphaned_files = []

    # Find all audio files
    audio_files = []
    for ext in config.AUDIO_EXTENSIONS:
        audio_files.extend(session_dir.glob(f"*{ext}"))

    # Filter out ignored files
    audio_files = [
        f for f in audio_files
        if f.name not in config.IGNORE_FILES
    ]

    # Match with JSON files
    for audio_file in audio_files:
        # Get base name without extension
        base_name = audio_file.stem

        # Look for matching JSON file
        json_file = session_dir / f"{base_name}{config.JSON_EXTENSION}"

        if json_file.exists():
            # Read timestamp from JSON
            try:
                with open(json_file, 'r') as f:
                    json_data = json.load(f)
                    timestamp_sec = json_data.get('video_timestamp_sec', None)

                paired_files.append({
                    'audio': audio_file,
                    'json': json_file,
                    'timestamp_sec': timestamp_sec,
                    'base_name': base_name
                })
            except (json.JSONDecodeError, KeyError) as e:
                print(f"Warning: Error reading {json_file}: {e}")
                orphaned_files.append(audio_file)
        else:
            orphaned_files.append(audio_file)

    # Sort paired files by timestamp
    paired_files.sort(key=lambda x: x['timestamp_sec'] if x['timestamp_sec'] is not None else float('inf'))

    # Sort orphaned files by creation time
    orphaned_files.sort(key=lambda x: x.stat().st_ctime)

    return paired_files, orphaned_files


def format_timestamp(seconds: float, include_milliseconds: bool = True) -> str:
    """
    Convert seconds to HH:MM:SS.mmm format.

    Args:
        seconds: Time in seconds
        include_milliseconds: Whether to include milliseconds

    Returns:
        Formatted timestamp string
    """
    td = timedelta(seconds=seconds)
    hours = int(td.total_seconds() // 3600)
    minutes = int((td.total_seconds() % 3600) // 60)
    secs = td.total_seconds() % 60

    if include_milliseconds:
        return f"{hours:02d}:{minutes:02d}:{secs:06.3f}"
    else:
        return f"{hours:02d}:{minutes:02d}:{int(secs):02d}"


def get_audio_duration(audio_file: Path) -> float:
    """
    Get the duration of an audio file in seconds using FFmpeg.

    Args:
        audio_file: Path to audio file (any format supported by FFmpeg)

    Returns:
        Duration in seconds
    """
    try:
        import subprocess
        import json

        # Use ffprobe to get duration (works with any audio format)
        cmd = [
            'ffprobe',
            '-v', 'quiet',
            '-print_format', 'json',
            '-show_format',
            str(audio_file)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        duration = float(data['format']['duration'])
        return duration

    except Exception as e:
        print(f"Warning: Could not get duration for {audio_file}: {e}")
        return 0.0


def count_words(text: str) -> int:
    """
    Count words in text.

    Args:
        text: Input text

    Returns:
        Number of words
    """
    return len(text.split())


def get_file_creation_time(file_path: Path) -> str:
    """
    Get file creation time as formatted string.

    Args:
        file_path: Path to file

    Returns:
        Formatted creation time
    """
    creation_time = os.path.getctime(file_path)
    dt = datetime.fromtimestamp(creation_time)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def detect_session_structure(input_dir: Path) -> Tuple[str, List[Dict]]:
    """
    Detect if input directory contains a single session or multiple sessions.

    Auto-detection logic:
    1. If input_dir contains .wav files directly → Single session
    2. If input_dir contains subdirectories with .wav files → Multiple sessions
    3. If no .wav files found → Error

    Args:
        input_dir: Path to input directory

    Returns:
        Tuple of (mode, sessions_list)
        - mode: "single" or "multiple"
        - sessions_list: List of session dicts with 'name', 'path', 'video_file' keys
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    # Check for audio files directly in input_dir
    audio_files_here = []
    for ext in config.AUDIO_EXTENSIONS:
        audio_files_here.extend(input_dir.glob(f"*{ext}"))

    # Filter out ignored files
    audio_files_here = [f for f in audio_files_here if f.name not in config.IGNORE_FILES]

    if audio_files_here:
        # Single session mode - audio files found directly
        video_file = None
        for ext in ['.mkv', '.mp4', '.avi', '.mov']:
            potential_videos = list(input_dir.glob(f"*{ext}"))
            if potential_videos:
                video_file = potential_videos[0]
                break

        session = {
            'name': input_dir.name,
            'path': input_dir,
            'video_file': video_file
        }
        return "single", [session]

    # No audio files at top level, check subdirectories
    sessions = []
    for item in input_dir.iterdir():
        if item.is_dir() and item.name not in [config.PROCESSED_DIR_NAME, config.OUTPUT_DIR_NAME]:
            # Check if this subfolder has audio files
            audio_in_subdir = []
            for ext in config.AUDIO_EXTENSIONS:
                audio_in_subdir.extend(item.glob(f"*{ext}"))

            audio_in_subdir = [f for f in audio_in_subdir if f.name not in config.IGNORE_FILES]

            if audio_in_subdir:
                # Found a session subfolder
                video_file = None
                for ext in ['.mkv', '.mp4', '.avi', '.mov']:
                    potential_videos = list(item.glob(f"*{ext}"))
                    if potential_videos:
                        video_file = potential_videos[0]
                        break

                sessions.append({
                    'name': item.name,
                    'path': item,
                    'video_file': video_file
                })

    if sessions:
        # Multiple sessions mode
        return "multiple", sessions

    # No audio files found anywhere
    raise FileNotFoundError(f"No audio files found in {input_dir} or its subdirectories")


def find_all_sessions(input_dir: Path) -> List[Dict]:
    """
    Find all session directories in the input directory.

    This is a legacy function kept for backwards compatibility.
    Use detect_session_structure() for new code.

    Args:
        input_dir: Path to input directory

    Returns:
        List of session info dicts with 'name', 'path', 'video_file' keys
    """
    try:
        mode, sessions = detect_session_structure(input_dir)
        return sessions
    except FileNotFoundError:
        return []


def ensure_dir(path: Path) -> None:
    """
    Create directory if it doesn't exist.

    Args:
        path: Directory path to create
    """
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def load_json_data(json_file: Path) -> Optional[Dict]:
    """
    Load and parse JSON file.

    Args:
        json_file: Path to JSON file

    Returns:
        Parsed JSON data or None if error
    """
    try:
        with open(json_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {json_file}: {e}")
        return None


def save_json_data(data: Dict, output_file: Path) -> bool:
    """
    Save data to JSON file with pretty formatting.

    Args:
        data: Data to save
        output_file: Output file path

    Returns:
        True if successful, False otherwise
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f"Error saving {output_file}: {e}")
        return False


def format_file_size(size_bytes: int) -> str:
    """
    Convert bytes to human-readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")

    # Test timestamp formatting
    print(f"\nTimestamp formatting:")
    print(f"  5.693 sec -> {format_timestamp(5.693)}")
    print(f"  65.5 sec -> {format_timestamp(65.5)}")
    print(f"  3661.25 sec -> {format_timestamp(3661.25)}")

    # Test finding sessions
    print(f"\nFinding sessions in {config.INPUT_DIR}...")
    sessions = find_all_sessions(config.INPUT_DIR)
    print(f"  Found {len(sessions)} session(s)")
    for session in sessions:
        print(f"    - {session['name']}")
        print(f"      Path: {session['path']}")
        print(f"      Video: {session['video_file']}")

    # Test finding audio/JSON pairs
    if sessions:
        session = sessions[0]
        print(f"\nFinding audio files in {session['name']}...")
        paired, orphaned = find_audio_json_pairs(session['path'])
        print(f"  Paired files: {len(paired)}")
        print(f"  Orphaned files: {len(orphaned)}")

        if paired:
            print(f"\n  First paired file:")
            print(f"    Audio: {paired[0]['audio'].name}")
            print(f"    JSON: {paired[0]['json'].name}")
            print(f"    Timestamp: {paired[0]['timestamp_sec']}s -> {format_timestamp(paired[0]['timestamp_sec'])}")

    print("\n✓ Utility functions tested successfully")
