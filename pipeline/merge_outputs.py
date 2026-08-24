"""
Output merging module for the transcription pipeline.

This module combines transcriptions with timestamps and generates
final output files (TXT and JSON formats).
"""

from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import math

import config
import utils


def analyze_segment_quality(segment: Dict) -> Dict:
    """
    Analyze quality metrics for a single segment.

    Args:
        segment: Whisper segment dictionary with quality metrics

    Returns:
        Dictionary with quality analysis results
    """
    compression_ratio = segment.get('compression_ratio', 1.0)
    no_speech_prob = segment.get('no_speech_prob', 0.0)
    avg_logprob = segment.get('avg_logprob', -0.5)

    # Detect quality issues
    likely_hallucination = compression_ratio > config.COMPRESSION_RATIO_THRESHOLD
    likely_silence = no_speech_prob > config.NO_SPEECH_THRESHOLD
    low_confidence = avg_logprob < config.CONFIDENCE_THRESHOLD

    # Calculate confidence score (0-1 scale from log probability)
    # avg_logprob typically ranges from 0 (perfect) to -inf (impossible)
    # Common range is -0.2 to -1.5
    confidence = math.exp(avg_logprob) if avg_logprob > -10 else 0.0

    return {
        'compression_ratio': round(compression_ratio, 2),
        'no_speech_prob': round(no_speech_prob, 3),
        'avg_logprob': round(avg_logprob, 3),
        'confidence': round(confidence, 3),
        'likely_hallucination': likely_hallucination,
        'likely_silence': likely_silence,
        'low_confidence': low_confidence,
        'quality_flags': []
    }


def add_quality_flags(quality_analysis: Dict) -> Dict:
    """
    Add human-readable quality flags to analysis.

    Args:
        quality_analysis: Quality analysis dictionary

    Returns:
        Updated dictionary with quality_flags list
    """
    flags = []

    if quality_analysis['likely_hallucination']:
        flags.append('hallucination_detected')
    if quality_analysis['likely_silence']:
        flags.append('silence_detected')
    if quality_analysis['low_confidence']:
        flags.append('low_confidence')

    quality_analysis['quality_flags'] = flags
    return quality_analysis


def format_diarization_as_text(diarization_result: List[Dict]) -> str:
    """Format diarization segments as readable speaker-attributed text."""
    if not diarization_result:
        return ""

    lines: List[str] = []
    previous_speaker = None

    for segment in diarization_result:
        speaker = segment.get("speaker", "UNKNOWN")
        text = segment.get("text", "").strip()
        if not text:
            continue

        if speaker != previous_speaker:
            if lines:
                lines.append("")
            lines.append(f"[{speaker}]: {text}")
        else:
            lines[-1] += f" {text}"

        previous_speaker = speaker

    return "\n".join(lines)


def get_diarization_for_file(diarization_results: Optional[Dict[str, List[Dict]]],
                             filename_stem: str) -> List[Dict]:
    """Return diarization segments for a file stem or an empty list."""
    if not diarization_results:
        return []
    return diarization_results.get(filename_stem, [])


def generate_individual_transcripts(paired_files: List[Dict],
                                    orphaned_files: List[Path],
                                    transcripts: Dict[str, Dict],
                                    output_dir: Path,
                                    diarization_results: Optional[Dict[str, List[Dict]]] = None) -> None:
    """
    Generate individual transcript files (TXT and JSON) with timestamp headers.

    Args:
        paired_files: List of paired audio/JSON file dicts
        orphaned_files: List of orphaned audio files
        transcripts: Dictionary mapping filename stems to Whisper result dicts
        output_dir: Output directory for transcript files
        diarization_results: Optional per-file diarization data
    """
    transcripts_dir = output_dir / "transcripts"
    utils.ensure_dir(transcripts_dir)

    # Process paired files
    for file_info in paired_files:
        audio_file = file_info['audio']
        timestamp_sec = file_info['timestamp_sec']
        filename_stem = audio_file.stem

        # Match by filename stem
        if filename_stem not in transcripts:
            continue

        result = transcripts[filename_stem]
        text = result["text"].strip()
        segments = result.get("segments", [])
        diarization = get_diarization_for_file(diarization_results, filename_stem)
        speaker_text = format_diarization_as_text(diarization)
        language = result.get("language", "unknown")
        duration = utils.get_audio_duration(audio_file)

        # Analyze quality for each segment
        segments_with_quality = []
        quality_flags = set()

        for segment in segments:
            quality = analyze_segment_quality(segment)
            quality = add_quality_flags(quality)

            # Collect overall quality flags
            quality_flags.update(quality['quality_flags'])

            segments_with_quality.append({
                "id": segment.get("id", 0),
                "start": round(segment.get("start", 0.0), 3),
                "end": round(segment.get("end", 0.0), 3),
                "text": segment.get("text", "").strip(),
                **quality
            })

        # Create TXT file with timestamp and quality warnings
        txt_file = transcripts_dir / f"{filename_stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            # Write timestamp header
            f.write(f"[VIDEO TIMESTAMP: {utils.format_timestamp(timestamp_sec)}]\n")

            # Write quality warnings if any
            if config.ENABLE_QUALITY_CHECKS and quality_flags:
                f.write(f"[QUALITY WARNINGS: {', '.join(sorted(quality_flags))}]\n")

            if diarization:
                speakers_detected = sorted({item.get("speaker", "UNKNOWN") for item in diarization})
                f.write(f"[SPEAKERS: {', '.join(speakers_detected)}]\n")

            f.write("\n")

            # Write transcription
            f.write(speaker_text if speaker_text else text)
            f.write("\n")

        # Create JSON file with metadata and quality metrics
        json_file = transcripts_dir / f"{filename_stem}.json"
        json_data = {
            "audio_file": audio_file.name,
            "has_video_timestamp": True,
            "video_timestamp_sec": timestamp_sec,
            "video_timestamp_formatted": utils.format_timestamp(timestamp_sec),
            "audio_duration_sec": round(duration, 3),
            "language": language,
            "transcription": text,
            "diarization_available": bool(diarization),
            "speakers_detected": sorted({item.get("speaker", "UNKNOWN") for item in diarization}),
            "diarization": diarization,
            "word_count": utils.count_words(text),
            "character_count": len(text),
            "quality_flags": sorted(list(quality_flags)),
            "segments": segments_with_quality if config.ENABLE_QUALITY_CHECKS else []
        }
        utils.save_json_data(json_data, json_file)

    # Process orphaned files
    for audio_file in orphaned_files:
        filename_stem = audio_file.stem

        # Match by filename stem
        if filename_stem not in transcripts:
            continue

        result = transcripts[filename_stem]
        text = result["text"].strip()
        segments = result.get("segments", [])
        diarization = get_diarization_for_file(diarization_results, filename_stem)
        speaker_text = format_diarization_as_text(diarization)
        language = result.get("language", "unknown")
        duration = utils.get_audio_duration(audio_file)
        creation_time = utils.get_file_creation_time(audio_file)

        # Analyze quality for each segment
        segments_with_quality = []
        quality_flags = set()

        for segment in segments:
            quality = analyze_segment_quality(segment)
            quality = add_quality_flags(quality)

            # Collect overall quality flags
            quality_flags.update(quality['quality_flags'])

            segments_with_quality.append({
                "id": segment.get("id", 0),
                "start": round(segment.get("start", 0.0), 3),
                "end": round(segment.get("end", 0.0), 3),
                "text": segment.get("text", "").strip(),
                **quality
            })

        # Create TXT file
        txt_file = transcripts_dir / f"{filename_stem}.txt"
        with open(txt_file, 'w', encoding='utf-8') as f:
            # Write orphan header
            f.write(f"[ORPHANED - NO VIDEO TIMESTAMP]\n")
            f.write(f"[File created: {creation_time}]\n")

            # Write quality warnings if any
            if config.ENABLE_QUALITY_CHECKS and quality_flags:
                f.write(f"[QUALITY WARNINGS: {', '.join(sorted(quality_flags))}]\n")

            if diarization:
                speakers_detected = sorted({item.get("speaker", "UNKNOWN") for item in diarization})
                f.write(f"[SPEAKERS: {', '.join(speakers_detected)}]\n")

            f.write("\n")
            f.write("This recording has no matching JSON timestamp file.\n\n")
            # Write transcription
            f.write(speaker_text if speaker_text else text)
            f.write("\n")

        # Create JSON file with metadata
        json_file = transcripts_dir / f"{filename_stem}.json"
        json_data = {
            "audio_file": audio_file.name,
            "has_video_timestamp": False,
            "file_created": creation_time,
            "audio_duration_sec": round(duration, 3),
            "language": language,
            "transcription": text,
            "diarization_available": bool(diarization),
            "speakers_detected": sorted({item.get("speaker", "UNKNOWN") for item in diarization}),
            "diarization": diarization,
            "word_count": utils.count_words(text),
            "character_count": len(text),
            "quality_flags": sorted(list(quality_flags)),
            "segments": segments_with_quality if config.ENABLE_QUALITY_CHECKS else [],
            "note": "No matching JSON timestamp file found"
        }
        utils.save_json_data(json_data, json_file)


def generate_combined_txt(paired_files: List[Dict],
                         orphaned_files: List[Path],
                         transcripts: Dict[str, Dict],
                         session_name: str,
                         video_file: Path,
                         output_file: Path,
                         diarization_results: Optional[Dict[str, List[Dict]]] = None) -> None:
    """
    Generate combined transcript in TXT format.

    Args:
        paired_files: List of paired audio/JSON file dicts
        orphaned_files: List of orphaned audio files
        transcripts: Dictionary mapping filename stems to Whisper result dicts
        session_name: Name of the session
        video_file: Path to video file (if exists)
        output_file: Output TXT file path
        diarization_results: Optional per-file diarization data
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Header
        f.write(config.HEADER_SEPARATOR + "\n")
        f.write("MICRO-PHENOMENOLOGICAL INTERVIEW TRANSCRIPT\n")
        f.write(config.HEADER_SEPARATOR + "\n")
        f.write(f"Session: {session_name}\n")
        if video_file:
            f.write(f"Video File: {video_file.name}\n")
        f.write(f"Total Recordings: {len(paired_files) + len(orphaned_files)}\n")
        f.write(f"Transcription Model: Whisper {config.WHISPER_MODEL}\n")
        f.write(f"Processing Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(config.HEADER_SEPARATOR + "\n\n\n")

        # Paired recordings
        for i, file_info in enumerate(paired_files, 1):
            audio_file = file_info['audio']
            timestamp_sec = file_info['timestamp_sec']
            filename_stem = audio_file.stem

            # Match by filename stem
            if filename_stem not in transcripts:
                continue

            result = transcripts[filename_stem]
            text = result["text"].strip()
            segments = result.get("segments", [])
            diarization = get_diarization_for_file(diarization_results, filename_stem)
            speaker_text = format_diarization_as_text(diarization)
            duration = utils.get_audio_duration(audio_file)

            # Analyze quality
            quality_flags = set()
            for segment in segments:
                quality = analyze_segment_quality(segment)
                quality = add_quality_flags(quality)
                quality_flags.update(quality['quality_flags'])

            f.write(config.SECTION_SEPARATOR + "\n")
            f.write(f"ANNOTATION #{i}\n")
            f.write(f"VIDEO TIMESTAMP: {utils.format_timestamp(timestamp_sec)} ({timestamp_sec} seconds)\n")
            f.write(f"AUDIO FILE: {audio_file.name}\n")
            f.write(f"DURATION: {duration:.1f} seconds\n")

            # Add quality warnings if any
            if config.ENABLE_QUALITY_CHECKS and quality_flags:
                f.write(f"QUALITY WARNINGS: {', '.join(sorted(quality_flags))}\n")

            if diarization:
                speakers_detected = sorted({item.get("speaker", "UNKNOWN") for item in diarization})
                f.write(f"SPEAKERS: {', '.join(speakers_detected)}\n")

            f.write(config.SECTION_SEPARATOR + "\n\n")
            f.write(speaker_text if speaker_text else text)
            f.write("\n\n\n")

        # Orphaned recordings (if any)
        if orphaned_files and any(f.stem in transcripts for f in orphaned_files):
            f.write(config.SECTION_SEPARATOR + "\n")
            f.write("ORPHANED RECORDINGS (No Video Timestamp)\n")
            f.write(config.SECTION_SEPARATOR + "\n\n")

            for audio_file in orphaned_files:
                filename_stem = audio_file.stem

                # Match by filename stem
                if filename_stem not in transcripts:
                    continue

                result = transcripts[filename_stem]
                text = result["text"].strip()
                diarization = get_diarization_for_file(diarization_results, filename_stem)
                speaker_text = format_diarization_as_text(diarization)
                creation_time = utils.get_file_creation_time(audio_file)
                duration = utils.get_audio_duration(audio_file)

                f.write(f"File: {audio_file.name}\n")
                f.write(f"Created: {creation_time}\n")
                f.write(f"Duration: {duration:.1f} seconds\n\n")
                f.write(speaker_text if speaker_text else text)
                f.write("\n\n")

        # Footer
        total_duration = sum(utils.get_audio_duration(f['audio']) for f in paired_files if f['audio'].stem in transcripts)
        total_words = sum(utils.count_words(transcripts[f['audio'].stem]["text"]) for f in paired_files if f['audio'].stem in transcripts)

        f.write(config.HEADER_SEPARATOR + "\n")
        f.write("END OF TRANSCRIPT\n")
        f.write(f"Total Audio Duration: {total_duration:.1f} seconds\n")
        f.write(f"Total Word Count: {total_words} words\n")
        f.write(config.HEADER_SEPARATOR + "\n")


def generate_combined_json(paired_files: List[Dict],
                          orphaned_files: List[Path],
                          transcripts: Dict[str, Dict],
                          session_name: str,
                          video_file: Path,
                          output_file: Path,
                          diarization_results: Optional[Dict[str, List[Dict]]] = None) -> None:
    """
    Generate combined transcript in JSON format.

    Args:
        paired_files: List of paired audio/JSON file dicts
        orphaned_files: List of orphaned audio files
        transcripts: Dictionary mapping filename stems to Whisper result dicts
        session_name: Name of the session
        video_file: Path to video file (if exists)
        output_file: Output JSON file path
        diarization_results: Optional per-file diarization data
    """
    # Build annotations list
    annotations = []
    total_quality_issues = {"hallucination": 0, "silence": 0, "low_confidence": 0}

    for i, file_info in enumerate(paired_files, 1):
        audio_file = file_info['audio']
        timestamp_sec = file_info['timestamp_sec']
        filename_stem = audio_file.stem

        # Match by filename stem
        if filename_stem not in transcripts:
            continue

        result = transcripts[filename_stem]
        text = result["text"].strip()
        segments = result.get("segments", [])
        diarization = get_diarization_for_file(diarization_results, filename_stem)
        language = result.get("language", "unknown")
        duration = utils.get_audio_duration(audio_file)

        # Analyze quality for each segment
        segments_with_quality = []
        quality_flags = set()

        for segment in segments:
            quality = analyze_segment_quality(segment)
            quality = add_quality_flags(quality)

            # Collect overall quality flags
            quality_flags.update(quality['quality_flags'])

            # Count quality issues
            if quality['likely_hallucination']:
                total_quality_issues["hallucination"] += 1
            if quality['likely_silence']:
                total_quality_issues["silence"] += 1
            if quality['low_confidence']:
                total_quality_issues["low_confidence"] += 1

            segments_with_quality.append({
                "id": segment.get("id", 0),
                "start": round(segment.get("start", 0.0), 3),
                "end": round(segment.get("end", 0.0), 3),
                "text": segment.get("text", "").strip(),
                **quality
            })

        annotations.append({
            "id": i,
            "has_video_timestamp": True,
            "video_timestamp_sec": timestamp_sec,
            "video_timestamp_formatted": utils.format_timestamp(timestamp_sec),
            "audio_file": audio_file.name,
            "audio_duration_sec": round(duration, 3),
            "language": language,
            "transcription": text,
            "diarization_available": bool(diarization),
            "speakers_detected": sorted({item.get("speaker", "UNKNOWN") for item in diarization}),
            "diarization": diarization,
            "word_count": utils.count_words(text),
            "character_count": len(text),
            "quality_flags": sorted(list(quality_flags)),
            "segments": segments_with_quality if config.ENABLE_QUALITY_CHECKS else []
        })

    # Build orphaned recordings list
    orphaned_recordings = []
    for audio_file in orphaned_files:
        filename_stem = audio_file.stem

        # Match by filename stem
        if filename_stem not in transcripts:
            continue

        result = transcripts[filename_stem]
        text = result["text"].strip()
        segments = result.get("segments", [])
        diarization = get_diarization_for_file(diarization_results, filename_stem)
        language = result.get("language", "unknown")
        duration = utils.get_audio_duration(audio_file)
        creation_time = utils.get_file_creation_time(audio_file)

        # Analyze quality for each segment
        segments_with_quality = []
        quality_flags = set()

        for segment in segments:
            quality = analyze_segment_quality(segment)
            quality = add_quality_flags(quality)

            quality_flags.update(quality['quality_flags'])

            # Count quality issues
            if quality['likely_hallucination']:
                total_quality_issues["hallucination"] += 1
            if quality['likely_silence']:
                total_quality_issues["silence"] += 1
            if quality['low_confidence']:
                total_quality_issues["low_confidence"] += 1

            segments_with_quality.append({
                "id": segment.get("id", 0),
                "start": round(segment.get("start", 0.0), 3),
                "end": round(segment.get("end", 0.0), 3),
                "text": segment.get("text", "").strip(),
                **quality
            })

        orphaned_recordings.append({
            "id": f"orphan_{len(orphaned_recordings) + 1}",
            "has_video_timestamp": False,
            "audio_file": audio_file.name,
            "file_created": creation_time,
            "audio_duration_sec": round(duration, 3),
            "language": language,
            "transcription": text,
            "diarization_available": bool(diarization),
            "speakers_detected": sorted({item.get("speaker", "UNKNOWN") for item in diarization}),
            "diarization": diarization,
            "word_count": utils.count_words(text),
            "quality_flags": sorted(list(quality_flags)),
            "segments": segments_with_quality if config.ENABLE_QUALITY_CHECKS else [],
            "note": "No matching JSON timestamp file found"
        })

    # Calculate statistics
    total_duration = sum(a['audio_duration_sec'] for a in annotations)
    total_words = sum(a['word_count'] for a in annotations)
    total_chars = sum(a['character_count'] for a in annotations)

    # Count recordings with quality issues
    recordings_with_hallucination = sum(1 for a in annotations if 'hallucination_detected' in a.get('quality_flags', []))
    recordings_with_silence = sum(1 for a in annotations if 'silence_detected' in a.get('quality_flags', []))
    recordings_with_low_confidence = sum(1 for a in annotations if 'low_confidence' in a.get('quality_flags', []))

    # Build complete JSON structure
    data = {
        "session_metadata": {
            "session_id": session_name,
            "video_file": video_file.name if video_file else None,
            "video_file_exists": video_file.exists() if video_file else False,
            "processing_timestamp": datetime.now().isoformat(),
            "transcription_model": f"whisper-{config.WHISPER_MODEL}",
            "total_recordings": len(paired_files) + len(orphaned_files),
            "total_with_timestamps": len(annotations),
            "total_orphaned": len(orphaned_recordings),
            "pipeline_version": config.PIPELINE_VERSION,
            "quality_checks_enabled": config.ENABLE_QUALITY_CHECKS
        },
        "annotations": annotations,
        "orphaned_recordings": orphaned_recordings,
        "statistics": {
            "total_audio_duration_sec": round(total_duration, 1),
            "total_audio_duration_formatted": utils.format_timestamp(total_duration, include_milliseconds=False),
            "total_words": total_words,
            "total_characters": total_chars,
            "average_annotation_duration_sec": round(total_duration / len(annotations), 1) if annotations else 0,
            "average_words_per_annotation": round(total_words / len(annotations)) if annotations else 0,
            "video_coverage": {
                "first_timestamp_sec": annotations[0]['video_timestamp_sec'] if annotations else None,
                "last_timestamp_sec": annotations[-1]['video_timestamp_sec'] if annotations else None,
                "span_sec": round(annotations[-1]['video_timestamp_sec'] - annotations[0]['video_timestamp_sec'], 3) if len(annotations) > 1 else 0,
                "span_formatted": utils.format_timestamp(annotations[-1]['video_timestamp_sec'] - annotations[0]['video_timestamp_sec'], include_milliseconds=False) if len(annotations) > 1 else "00:00:00"
            },
            "quality_metrics": {
                "total_segments": sum(len(a.get('segments', [])) for a in annotations),
                "segments_with_hallucination": total_quality_issues["hallucination"],
                "segments_with_silence": total_quality_issues["silence"],
                "segments_with_low_confidence": total_quality_issues["low_confidence"],
                "recordings_with_hallucination": recordings_with_hallucination,
                "recordings_with_silence": recordings_with_silence,
                "recordings_with_low_confidence": recordings_with_low_confidence,
                "thresholds": {
                    "compression_ratio_threshold": config.COMPRESSION_RATIO_THRESHOLD,
                    "no_speech_threshold": config.NO_SPEECH_THRESHOLD,
                    "confidence_threshold": config.CONFIDENCE_THRESHOLD
                }
            } if config.ENABLE_QUALITY_CHECKS else {}
        }
    }

    # Save JSON
    utils.save_json_data(data, output_file)


def generate_plain_text(paired_files: List[Dict],
                       orphaned_files: List[Path],
                       transcripts: Dict[str, Dict],
                       output_file: Path,
                       diarization_results: Optional[Dict[str, List[Dict]]] = None) -> None:
    """
    Generate plain text transcript without timestamps or metadata.

    Args:
        paired_files: List of paired audio/JSON file dicts
        orphaned_files: List of orphaned audio files
        transcripts: Dictionary mapping filename stems to Whisper result dicts
        output_file: Output TXT file path
        diarization_results: Optional per-file diarization data
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        # Process paired files (ordered by timestamp)
        for i, file_info in enumerate(paired_files):
            audio_file = file_info['audio']
            filename_stem = audio_file.stem

            # Match by filename stem
            if filename_stem not in transcripts:
                continue

            result = transcripts[filename_stem]
            text = result["text"].strip()
            diarization = get_diarization_for_file(diarization_results, filename_stem)
            speaker_text = format_diarization_as_text(diarization)

            # Write transcription text
            f.write(speaker_text if speaker_text else text)
            
            # Add double newline separator between recordings
            if i < len(paired_files) - 1 or orphaned_files:
                f.write("\n\n")

        # Process orphaned files
        for i, audio_file in enumerate(orphaned_files):
            filename_stem = audio_file.stem

            # Match by filename stem
            if filename_stem not in transcripts:
                continue

            result = transcripts[filename_stem]
            text = result["text"].strip()
            diarization = get_diarization_for_file(diarization_results, filename_stem)
            speaker_text = format_diarization_as_text(diarization)

            # Write transcription text
            f.write(speaker_text if speaker_text else text)
            
            # Add double newline separator between recordings (except last)
            if i < len(orphaned_files) - 1:
                f.write("\n\n")


if __name__ == "__main__":
    print("Output merge module loaded successfully")
    print("This module is meant to be called from run_pipeline.py")
