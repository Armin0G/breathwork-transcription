"""
Speaker diarization module for the breathwork transcription pipeline.

This module wraps the whisper-diarization reference logic into a reusable class
that can be called from the main pipeline without script-style side effects.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import faster_whisper
import torch
from ctc_forced_aligner import (
    generate_emissions,
    get_alignments,
    get_spans,
    load_alignment_model,
    postprocess_results,
    preprocess_text,
)
from deepmultilingualpunctuation import PunctuationModel
import nltk

import config

# Languages supported by punctuate-all in the reference project.
PUNCT_MODEL_LANGS = {
    "en", "fr", "de", "es", "it", "nl", "pt", "bg", "pl", "cs", "sk", "sl"
}

# Whisper language code to ISO-639-2 mapping expected by ctc-forced-aligner.
LANGS_TO_ISO = {
    "af": "afr", "am": "amh", "ar": "ara", "as": "asm", "az": "aze", "ba": "bak",
    "be": "bel", "bg": "bul", "bn": "ben", "bo": "tib", "br": "bre", "bs": "bos",
    "ca": "cat", "cs": "cze", "cy": "wel", "da": "dan", "de": "ger", "el": "gre",
    "en": "eng", "es": "spa", "et": "est", "eu": "baq", "fa": "per", "fi": "fin",
    "fo": "fao", "fr": "fre", "gl": "glg", "gu": "guj", "ha": "hau", "haw": "haw",
    "he": "heb", "hi": "hin", "hr": "hrv", "ht": "hat", "hu": "hun", "hy": "arm",
    "id": "ind", "is": "ice", "it": "ita", "ja": "jpn", "jw": "jav", "ka": "geo",
    "kk": "kaz", "km": "khm", "kn": "kan", "ko": "kor", "la": "lat", "lb": "ltz",
    "ln": "lin", "lo": "lao", "lt": "lit", "lv": "lav", "mg": "mlg", "mi": "mao",
    "mk": "mac", "ml": "mal", "mn": "mon", "mr": "mar", "ms": "may", "mt": "mlt",
    "my": "bur", "ne": "nep", "nl": "dut", "nn": "nno", "no": "nor", "oc": "oci",
    "pa": "pan", "pl": "pol", "ps": "pus", "pt": "por", "ro": "rum", "ru": "rus",
    "sa": "san", "sd": "snd", "si": "sin", "sk": "slo", "sl": "slv", "sn": "sna",
    "so": "som", "sq": "alb", "sr": "srp", "su": "sun", "sv": "swe", "sw": "swa",
    "ta": "tam", "te": "tel", "tg": "tgk", "th": "tha", "tk": "tuk", "tl": "tgl",
    "tr": "tur", "tt": "tat", "uk": "ukr", "ur": "urd", "uz": "uzb", "vi": "vie",
    "yi": "yid", "yo": "yor", "yue": "yue", "zh": "chi",
}


class DiarizationProcessor:
    """Reusable processor for forced alignment + NeMo MSDD diarization."""

    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.alignment_model = None
        self.alignment_tokenizer = None
        self.punct_model = None

    def process(self, audio_file: Path, transcript_result: Dict) -> Optional[List[Dict]]:
        """
        Run diarization for one audio file.

        Args:
            audio_file: Path to preprocessed audio file.
            transcript_result: Transcription result from transcribe.py.

        Returns:
            List of sentence dictionaries with speaker labels, or None on failure.
        """
        try:
            if not audio_file.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_file}")

            full_text = (transcript_result.get("text") or "").strip()
            if not full_text:
                return None

            language = transcript_result.get("language", config.LANGUAGE)
            self._load_alignment_model()

            audio_waveform = faster_whisper.decode_audio(str(audio_file))
            if config.ENABLE_STEMMING:
                audio_waveform = self._apply_stemming(audio_file, audio_waveform)

            word_timestamps = self._forced_align(audio_waveform, full_text, language)
            speaker_timestamps = self._run_msdd_diarization(audio_waveform)
            if not speaker_timestamps:
                return None

            words_speakers = self._get_words_speaker_mapping(
                word_timestamps,
                speaker_timestamps,
            )

            if language in PUNCT_MODEL_LANGS:
                words_speakers = self._restore_punctuation(words_speakers)

            words_speakers = self._realign_with_punctuation(words_speakers)
            sentence_mapping = self._get_sentences_speaker_mapping(
                words_speakers,
                speaker_timestamps,
            )

            results: List[Dict] = []
            for sentence in sentence_mapping:
                raw_speaker = sentence["speaker"]
                speaker_num = int(raw_speaker.split()[-1])
                results.append(
                    {
                        "speaker": config.SPEAKER_LABEL_FORMAT.format(speaker_num),
                        "start_time_ms": int(sentence["start_time"]),
                        "end_time_ms": int(sentence["end_time"]),
                        "text": sentence["text"].strip(),
                    }
                )

            return results

        except Exception as exc:
            print(f"  [diarization] failed for {audio_file.name}: {exc}")
            return None

    def _load_alignment_model(self) -> None:
        if self.alignment_model is not None and self.alignment_tokenizer is not None:
            return

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.alignment_model, self.alignment_tokenizer = load_alignment_model(
            self.device,
            dtype=dtype,
        )

    def _load_punctuation_model(self) -> None:
        if self.punct_model is None:
            self.punct_model = PunctuationModel(model="kredor/punctuate-all")

    def _forced_align(self, audio_waveform, transcript_text: str, language: str) -> List[Dict]:
        batch_size = max(1, int(config.WHISPER_BATCH_SIZE))
        emissions, stride = generate_emissions(
            self.alignment_model,
            torch.from_numpy(audio_waveform)
            .to(self.alignment_model.dtype)
            .to(self.alignment_model.device),
            batch_size=batch_size,
        )

        iso_language = LANGS_TO_ISO.get(language, "eng")
        tokens_starred, text_starred = preprocess_text(
            transcript_text,
            romanize=True,
            language=iso_language,
        )

        segments, scores, blank_token = get_alignments(
            emissions,
            tokens_starred,
            self.alignment_tokenizer,
        )
        spans = get_spans(tokens_starred, segments, blank_token)
        return postprocess_results(text_starred, spans, stride, scores)

    def _run_msdd_diarization(self, audio_waveform) -> Optional[List[Tuple[int, int, int]]]:
        if getattr(config, "DIARIZATION_REPO_PATH", None) is not None:
            diarization_repo = Path(config.DIARIZATION_REPO_PATH).resolve()
        else:
            # Default: whisper-diarization as sibling of breathwork-transcription
            repo_root = Path(__file__).resolve().parents[1].parent  # breathwork-transcription parent
            diarization_repo = repo_root / "whisper-diarization"
        if not diarization_repo.exists():
            raise FileNotFoundError(
                "whisper-diarization repository not found. "
                f"Expected at: {diarization_repo}. "
                "Clone it as a sibling of breathwork-transcription or set config.DIARIZATION_REPO_PATH."
            )

        if str(diarization_repo) not in sys.path:
            sys.path.insert(0, str(diarization_repo))

        from diarization import MSDDDiarizer

        diarizer = MSDDDiarizer(device=self.device)
        try:
            labels = diarizer.diarize(torch.from_numpy(audio_waveform).unsqueeze(0))
            return labels
        finally:
            del diarizer
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def _apply_stemming(self, audio_file: Path, audio_waveform):
        with tempfile.TemporaryDirectory() as tmp_dir:
            command = [
                sys.executable,
                "-m",
                "demucs.separate",
                "-n",
                "htdemucs",
                "--two-stems=vocals",
                str(audio_file),
                "-o",
                tmp_dir,
                "--device",
                self.device,
            ]
            proc = subprocess.run(command, capture_output=True, text=True)
            if proc.returncode != 0:
                return audio_waveform

            vocals_path = Path(tmp_dir) / "htdemucs" / audio_file.stem / "vocals.wav"
            if vocals_path.exists():
                return faster_whisper.decode_audio(str(vocals_path))
            return audio_waveform

    @staticmethod
    def _get_word_ts_anchor(start_ms: int, end_ms: int, option: str = "start") -> float:
        if option == "end":
            return end_ms
        if option == "mid":
            return (start_ms + end_ms) / 2
        return start_ms

    def _get_words_speaker_mapping(
        self,
        word_timestamps: List[Dict],
        speaker_timestamps: List[Tuple[int, int, int]],
        anchor_option: str = "start",
    ) -> List[Dict]:
        s, e, speaker = speaker_timestamps[0]
        turn_idx = 0
        mapping: List[Dict] = []

        for word_entry in word_timestamps:
            ws = int(word_entry["start"] * 1000)
            we = int(word_entry["end"] * 1000)
            word = word_entry["text"]
            anchor = self._get_word_ts_anchor(ws, we, option=anchor_option)

            while anchor > float(e):
                turn_idx = min(turn_idx + 1, len(speaker_timestamps) - 1)
                s, e, speaker = speaker_timestamps[turn_idx]
                if turn_idx == len(speaker_timestamps) - 1:
                    e = int(self._get_word_ts_anchor(ws, we, option="end"))

            mapping.append(
                {
                    "word": word,
                    "start_time": ws,
                    "end_time": we,
                    "speaker": speaker,
                }
            )

        return mapping

    def _restore_punctuation(self, words_speakers: List[Dict]) -> List[Dict]:
        self._load_punctuation_model()

        words = [item["word"] for item in words_speakers]
        labeled_words = self.punct_model.predict(words, chunk_size=230)

        ending_puncts = ".?!"
        model_puncts = ".,;:!?"
        is_acronym = lambda token: re.fullmatch(r"\b(?:[a-zA-Z]\.){2,}", token)

        for word_dict, labeled_tuple in zip(words_speakers, labeled_words):
            token = word_dict["word"]
            punct = labeled_tuple[1]
            if token and punct in ending_puncts and (token[-1] not in model_puncts or is_acronym(token)):
                token += punct
                if token.endswith(".."):
                    token = token.rstrip(".")
                word_dict["word"] = token

        return words_speakers

    def _realign_with_punctuation(
        self,
        word_speaker_mapping: List[Dict],
        max_words_in_sentence: int = 50,
    ) -> List[Dict]:
        sentence_end_punct = ".?!"

        def is_sentence_end(idx: int) -> bool:
            return idx >= 0 and word_speaker_mapping[idx]["word"][-1] in sentence_end_punct

        words = [entry["word"] for entry in word_speaker_mapping]
        speakers = [entry["speaker"] for entry in word_speaker_mapping]

        def first_word_idx(word_idx: int) -> int:
            left = word_idx
            while (
                left > 0
                and word_idx - left < max_words_in_sentence
                and speakers[left - 1] == speakers[left]
                and not is_sentence_end(left - 1)
            ):
                left -= 1
            return left if left == 0 or is_sentence_end(left - 1) else -1

        def last_word_idx(word_idx: int, max_words: int) -> int:
            right = word_idx
            while (
                right < len(words) - 1
                and right - word_idx < max_words
                and not is_sentence_end(right)
            ):
                right += 1
            return right if right == len(words) - 1 or is_sentence_end(right) else -1

        idx = 0
        total = len(word_speaker_mapping)
        while idx < total:
            if idx < total - 1 and speakers[idx] != speakers[idx + 1] and not is_sentence_end(idx):
                left_idx = first_word_idx(idx)
                right_idx = (
                    last_word_idx(idx, max_words_in_sentence - idx + left_idx - 1)
                    if left_idx > -1
                    else -1
                )
                if min(left_idx, right_idx) == -1:
                    idx += 1
                    continue

                labels = speakers[left_idx:right_idx + 1]
                dominant = max(set(labels), key=labels.count)
                if labels.count(dominant) < len(labels) // 2:
                    idx += 1
                    continue

                speakers[left_idx:right_idx + 1] = [dominant] * (right_idx - left_idx + 1)
                idx = right_idx
            idx += 1

        output = []
        for i, entry in enumerate(word_speaker_mapping):
            patched = entry.copy()
            patched["speaker"] = speakers[i]
            output.append(patched)
        return output

    def _get_sentences_speaker_mapping(
        self,
        word_speaker_mapping: List[Dict],
        speaker_timestamps: List[Tuple[int, int, int]],
    ) -> List[Dict]:
        sentence_checker = nltk.tokenize.PunktSentenceTokenizer().text_contains_sentbreak

        s, e, speaker = speaker_timestamps[0]
        prev_speaker = speaker
        sentence = {
            "speaker": f"Speaker {speaker}",
            "start_time": s,
            "end_time": e,
            "text": "",
        }
        sentences = []

        for word_entry in word_speaker_mapping:
            word = word_entry["word"]
            speaker = word_entry["speaker"]
            s = word_entry["start_time"]
            e = word_entry["end_time"]

            if speaker != prev_speaker or sentence_checker(sentence["text"] + " " + word):
                sentences.append(sentence)
                sentence = {
                    "speaker": f"Speaker {speaker}",
                    "start_time": s,
                    "end_time": e,
                    "text": "",
                }
            else:
                sentence["end_time"] = e

            sentence["text"] += word + " "
            prev_speaker = speaker

        sentences.append(sentence)
        return sentences
