#!/usr/bin/env python3
"""
Main pipeline script for micro-phenomenological interview transcription.

This script orchestrates the complete transcription workflow:
1. Find session folders and audio/JSON pairs
2. Preprocess audio (normalize, convert format)
3. Transcribe using Whisper
3b. Speaker diarization (optional)
4. Merge transcriptions with timestamps
5. Generate output files (TXT, JSON, report)
6. Clean up intermediate files

Usage:
    python run_pipeline.py                    # Process all sessions
    python run_pipeline.py --session NAME     # Process specific session
"""

import argparse
import shutil
from pathlib import Path
from datetime import datetime
import time

import config
import utils
import preprocess_audio
import transcribe
import merge_outputs


def process_session(session_info: dict, cleanup: bool = True,
                    enable_diarization: bool = True) -> bool:
    """
    Process a single session.

    Args:
        session_info: Session information dict from utils.detect_session_structure()
        cleanup: Whether to delete intermediate files after processing
        enable_diarization: Whether to run speaker diarization (can be overridden by --no-diarization)

    Returns:
        True if successful, False otherwise
    """
    session_name = session_info['name']
    session_path = session_info['path']
    video_file = session_info['video_file']

    print(f"\n{'=' * 80}")
    print(f"Processing session: {session_name}")
    print(f"{'=' * 80}\n")

    start_time = time.time()

    # Step 1: Find audio files and match with JSON
    print("Step 1: Finding audio files and matching with timestamps...")
    paired_files, orphaned_files = utils.find_audio_json_pairs(session_path)

    print(f"  ✓ Found {len(paired_files)} files with timestamps")
    if orphaned_files:
        print(f"  ⚠ Found {len(orphaned_files)} orphaned files (no JSON timestamp)")

    if not paired_files and not orphaned_files:
        print("  ✗ No audio files found in session!")
        return False

    all_audio_files = [f['audio'] for f in paired_files] + orphaned_files

    # Step 2: Preprocess audio
    print("\nStep 2: Preprocessing audio files...")
    # Create processed audio dir inside the session folder
    normalized_dir = session_path / config.PROCESSED_DIR_NAME

    preprocessed_files = preprocess_audio.preprocess_audio_files(
        all_audio_files,
        normalized_dir,
        parallel=True
    )

    if not preprocessed_files:
        print("  ✗ Audio preprocessing failed!")
        return False

    # Step 3: Transcribe with Whisper
    print("\nStep 3: Transcribing audio with Whisper...")
    # Create output dir inside the session folder
    output_dir = session_path / config.OUTPUT_DIR_NAME
    utils.ensure_dir(output_dir)

    transcriber = transcribe.WhisperTranscriber(config.WHISPER_MODEL)
    transcripts = transcriber.transcribe_files(preprocessed_files, output_dir)

    if not transcripts:
        print("  ✗ Transcription failed!")
        return False

    # Step 3b: Speaker diarization (optional)
    diarization_results = {}
    if enable_diarization and config.ENABLE_DIARIZATION:
        print("\nStep 3b: Speaker diarization...")
        from diarize import DiarizationProcessor

        diarizer = DiarizationProcessor()
        all_audio_for_diarization = [f["audio"] for f in paired_files] + orphaned_files

        for audio_file in all_audio_for_diarization:
            filename_stem = audio_file.stem
            if filename_stem not in transcripts:
                continue

            preprocessed_audio = normalized_dir / audio_file.name
            if not preprocessed_audio.exists():
                print(f"  ⚠ Skipping {audio_file.name}: preprocessed file not found")
                continue

            print(f"  Diarizing: {audio_file.name}")
            result = diarizer.process(
                audio_file=preprocessed_audio,
                transcript_result=transcripts[filename_stem],
            )

            if result:
                diarization_results[filename_stem] = result
                speakers = sorted({s["speaker"] for s in result})
                print(f"  ✓ {len(speakers)} speaker(s): {', '.join(speakers)}")
            else:
                print(f"  ⚠ Diarization failed for {audio_file.name}")

        print(f"\n  ✓ Diarization done: {len(diarization_results)}/{len(all_audio_for_diarization)} files")
    else:
        if not enable_diarization:
            print("\nStep 3b: Diarization disabled (--no-diarization)")
        else:
            print("\nStep 3b: Diarization disabled (ENABLE_DIARIZATION=False)")

    # Step 4: Generate individual transcript files
    if config.GENERATE_INDIVIDUAL_TRANSCRIPTS:
        print("\nStep 4: Generating individual transcript files...")
        merge_outputs.generate_individual_transcripts(
            paired_files,
            orphaned_files,
            transcripts,
            output_dir,
            diarization_results=diarization_results,
        )
        print(f"  ✓ Individual transcripts saved to: {output_dir / 'transcripts'}")

    # Step 5: Generate combined TXT
    if config.GENERATE_COMBINED_TXT:
        print("\nStep 5: Generating combined transcript (TXT)...")
        txt_file = output_dir / "combined_transcript.txt"
        merge_outputs.generate_combined_txt(
            paired_files,
            orphaned_files,
            transcripts,
            session_name,
            video_file,
            txt_file,
            diarization_results=diarization_results,
        )
        print(f"  ✓ Combined TXT saved to: {txt_file}")

    # Step 6: Generate combined JSON
    if config.GENERATE_COMBINED_JSON:
        print("\nStep 6: Generating combined transcript (JSON)...")
        json_file = output_dir / "combined_transcript.json"
        merge_outputs.generate_combined_json(
            paired_files,
            orphaned_files,
            transcripts,
            session_name,
            video_file,
            json_file,
            diarization_results=diarization_results,
        )
        print(f"  ✓ Combined JSON saved to: {json_file}")

    # Step 7: Generate plain text transcript
    if config.GENERATE_PLAIN_TEXT:
        print("\nStep 7: Generating plain text transcript...")
        plain_text_file = output_dir / "plain_text_transcript.txt"
        merge_outputs.generate_plain_text(
            paired_files,
            orphaned_files,
            transcripts,
            plain_text_file,
            diarization_results=diarization_results,
        )
        print(f"  ✓ Plain text saved to: {plain_text_file}")

    # Step 8: Generate processing report
    if config.GENERATE_PROCESSING_REPORT:
        print("\nStep 8: Generating processing report...")
        report_file = output_dir / "processing_report.txt"
        generate_processing_report(
            session_name,
            video_file,
            paired_files,
            orphaned_files,
            transcripts,
            start_time,
            report_file
        )
        print(f"  ✓ Processing report saved to: {report_file}")

    # Step 9: Cleanup intermediate files
    if cleanup and config.DELETE_INTERMEDIATE_FILES:
        print("\nStep 9: Cleaning up intermediate files...")
        if normalized_dir.exists():
            shutil.rmtree(normalized_dir)
            print(f"  ✓ Deleted: {normalized_dir}")

    # Done!
    elapsed_time = time.time() - start_time
    print(f"\n{'=' * 80}")
    print(f"✓ Session processed successfully in {elapsed_time:.1f} seconds")
    print(f"{'=' * 80}\n")

    return True


def generate_processing_report(session_name: str, video_file: Path,
                               paired_files: list, orphaned_files: list,
                               transcripts: dict, start_time: float,
                               output_file: Path) -> None:
    """Generate processing report."""
    elapsed = time.time() - start_time

    total_files = len(paired_files) + len(orphaned_files)
    successful = len(transcripts)
    total_duration = sum(utils.get_audio_duration(f['audio']) for f in paired_files if f['audio'].stem in transcripts)
    total_words = sum(utils.count_words(transcripts[f['audio'].stem]["text"]) for f in paired_files if f['audio'].stem in transcripts)
    total_chars = sum(len(transcripts[f['audio'].stem]["text"]) for f in paired_files if f['audio'].stem in transcripts)

    # Calculate quality statistics
    total_segments = 0
    segments_with_hallucination = 0
    segments_with_silence = 0
    segments_with_low_confidence = 0
    recordings_with_issues = 0

    for file_stem in transcripts:
        result = transcripts[file_stem]
        segments = result.get("segments", [])
        total_segments += len(segments)

        has_quality_issue = False
        for segment in segments:
            from merge_outputs import analyze_segment_quality, add_quality_flags
            quality = analyze_segment_quality(segment)
            quality = add_quality_flags(quality)

            if quality['likely_hallucination']:
                segments_with_hallucination += 1
                has_quality_issue = True
            if quality['likely_silence']:
                segments_with_silence += 1
                has_quality_issue = True
            if quality['low_confidence']:
                segments_with_low_confidence += 1
                has_quality_issue = True

        if has_quality_issue:
            recordings_with_issues += 1

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("TRANSCRIPTION PROCESSING REPORT\n")
        f.write("=" * 80 + "\n")
        f.write(f"Session: {session_name}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: Whisper {config.WHISPER_MODEL} (244M parameters)\n")
        f.write("=" * 80 + "\n\n")

        f.write("FILES PROCESSED\n")
        f.write("─" * 80 + "\n")
        f.write(f"Total audio files found:        {total_files}\n")
        f.write(f"With JSON timestamps:           {len(paired_files)}\n")
        f.write(f"Orphaned (no JSON):             {len(orphaned_files)}\n")
        f.write(f"Successfully transcribed:       {successful}\n")
        f.write(f"Failed transcriptions:          {total_files - successful}\n")
        f.write(f"Success rate:                   {100 * successful / total_files:.1f}%\n\n")

        f.write("TIMING\n")
        f.write("─" * 80 + "\n")
        f.write(f"Total processing time:          {int(elapsed // 60)}m {int(elapsed % 60)}s\n")
        if successful > 0:
            f.write(f"Average per file:               {elapsed / successful:.1f}s\n")
        if total_duration > 0:
            f.write(f"Audio duration / processing:    {total_duration / elapsed:.2f}x (realtime)\n\n")

        f.write("CONTENT STATISTICS\n")
        f.write("─" * 80 + "\n")
        f.write(f"Total audio duration:           {total_duration:.1f} seconds\n")
        f.write(f"Total words transcribed:        {total_words}\n")
        f.write(f"Total characters:               {total_chars:,}\n")
        if successful > 0:
            f.write(f"Average words per recording:    {total_words // successful}\n")

        if paired_files:
            durations = [utils.get_audio_duration(f['audio']) for f in paired_files]
            f.write(f"Shortest recording:             {min(durations):.1f}s ({paired_files[durations.index(min(durations))]['audio'].name})\n")
            f.write(f"Longest recording:              {max(durations):.1f}s ({paired_files[durations.index(max(durations))]['audio'].name})\n\n")

        if paired_files and len(paired_files) > 0:
            f.write("VIDEO COVERAGE\n")
            f.write("─" * 80 + "\n")
            first_ts = paired_files[0]['timestamp_sec']
            last_ts = paired_files[-1]['timestamp_sec']
            f.write(f"First annotation timestamp:     {utils.format_timestamp(first_ts)}\n")
            f.write(f"Last annotation timestamp:      {utils.format_timestamp(last_ts)}\n")
            f.write(f"Total video span covered:       {utils.format_timestamp(last_ts - first_ts, include_milliseconds=False)} ({last_ts - first_ts:.1f} seconds)\n\n")

        f.write("OUTPUT FILES CREATED\n")
        f.write("─" * 80 + "\n")
        f.write(f"✓ {successful} individual transcript files (transcripts/*.txt)\n")
        if config.GENERATE_COMBINED_TXT:
            f.write(f"✓ combined_transcript.txt\n")
        if config.GENERATE_COMBINED_JSON:
            f.write(f"✓ combined_transcript.json\n")
        if config.GENERATE_PLAIN_TEXT:
            f.write(f"✓ plain_text_transcript.txt\n")
        f.write(f"✓ processing_report.txt\n\n")

        f.write("SYSTEM INFO\n")
        f.write("─" * 80 + "\n")
        import platform
        import sys
        f.write(f"Operating System: {platform.system()} {platform.release()}\n")
        f.write(f"Python Version: {sys.version.split()[0]}\n")
        f.write(f"Whisper Model: {config.WHISPER_MODEL}\n")
        f.write(f"Pipeline Version: {config.PIPELINE_VERSION}\n\n")

        if config.ENABLE_QUALITY_CHECKS and total_segments > 0:
            f.write("QUALITY METRICS\n")
            f.write("─" * 80 + "\n")
            f.write(f"Total segments analyzed:        {total_segments}\n")
            f.write(f"Segments with hallucination:    {segments_with_hallucination} ({100 * segments_with_hallucination / total_segments:.1f}%)\n")
            f.write(f"Segments with silence:          {segments_with_silence} ({100 * segments_with_silence / total_segments:.1f}%)\n")
            f.write(f"Segments with low confidence:   {segments_with_low_confidence} ({100 * segments_with_low_confidence / total_segments:.1f}%)\n")
            f.write(f"Recordings with quality issues: {recordings_with_issues} ({100 * recordings_with_issues / successful:.1f}%)\n\n")

            f.write("QUALITY THRESHOLDS\n")
            f.write("─" * 80 + "\n")
            f.write(f"Compression ratio threshold:    {config.COMPRESSION_RATIO_THRESHOLD} (hallucination detector)\n")
            f.write(f"No-speech probability threshold: {config.NO_SPEECH_THRESHOLD} (silence detector)\n")
            f.write(f"Confidence threshold:           {config.CONFIDENCE_THRESHOLD} (low confidence)\n\n")

        f.write("QUALITY NOTES\n")
        f.write("─" * 80 + "\n")
        f.write("- All transcriptions include verbatim speech (filler words preserved)\n")
        f.write(f"- Temperature: {config.TEMPERATURE} (deterministic, reproducible)\n")
        f.write(f"- Language: {config.LANGUAGE} (forced)\n")
        f.write("- No post-processing or correction applied\n")
        if config.ENABLE_QUALITY_CHECKS:
            f.write("- Quality checks enabled (hallucination & silence detection)\n")
        f.write("\n")

        f.write("=" * 80 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 80 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transcribe micro-phenomenological interview recordings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single session folder
  python3 run_pipeline.py --input /path/to/session_folder

  # Process multiple session folders
  python3 run_pipeline.py --input /path/to/folder_with_sessions

  # Use current directory
  python3 run_pipeline.py --input .

  # Process specific session by name (when using multiple sessions)
  python3 run_pipeline.py --input /path/to/sessions --session session1

Output is always placed in a 'transcripts/' subfolder within each session folder.
        """
    )
    parser.add_argument(
        '--input',
        type=str,
        default='.',
        help="Input directory containing audio files (default: current directory)"
    )
    parser.add_argument(
        '--session',
        type=str,
        help="Process specific session by name (for multiple sessions mode)"
    )
    parser.add_argument(
        '--no-cleanup',
        action='store_true',
        help="Keep intermediate processed audio files"
    )
    parser.add_argument(
        '--no-diarization',
        action='store_true',
        help="Disable speaker diarization (transcription only)"
    )

    args = parser.parse_args()

    # Validate configuration
    try:
        config.validate_config()
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return 1

    # Resolve input directory
    input_dir = Path(args.input).resolve()

    # Detect session structure
    print(f"\nScanning input directory: {input_dir}")
    try:
        mode, sessions = utils.detect_session_structure(input_dir)
    except (FileNotFoundError, NotADirectoryError) as e:
        print(f"✗ {e}")
        return 1

    # Print detection results
    if mode == "single":
        print(f"✓ Detected: Single session")
        audio_count = len(list(sessions[0]['path'].glob('*.wav'))) + len(list(sessions[0]['path'].glob('*.WAV')))
        print(f"  Audio files found: {audio_count}")
    else:
        print(f"✓ Detected: Multiple sessions ({len(sessions)} folders)")
        for session in sessions:
            audio_count = len(list(session['path'].glob('*.wav'))) + len(list(session['path'].glob('*.WAV')))
            print(f"  - {session['name']} ({audio_count} audio files)")

    # Filter by session name if specified
    if args.session:
        sessions = [s for s in sessions if s['name'] == args.session]
        if not sessions:
            print(f"✗ Session '{args.session}' not found!")
            return 1
        print(f"\nFiltering to session: {args.session}")

    # Process each session
    cleanup = not args.no_cleanup
    enable_diarization = not args.no_diarization
    success_count = 0

    for session in sessions:
        try:
            if process_session(session, cleanup=cleanup,
                               enable_diarization=enable_diarization):
                success_count += 1
        except Exception as e:
            print(f"\n✗ Error processing session {session['name']}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"PROCESSING COMPLETE")
    print(f"{'=' * 80}")
    print(f"Successfully processed: {success_count}/{len(sessions)} sessions")
    if sessions:
        print(f"\nOutput locations:")
        for session in sessions:
            output_path = session['path'] / config.OUTPUT_DIR_NAME
            print(f"  - {output_path}")
    print(f"{'=' * 80}\n")

    return 0 if success_count == len(sessions) else 1


if __name__ == "__main__":
    exit(main())
