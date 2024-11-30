from concurrent.futures import ThreadPoolExecutor

from runpod.serverless.utils import rp_cuda
from pydantic import BaseModel
from typing import List
from mutagen.wave import WAVE
from mutagen.mp3 import MP3
from mutagen.mp4 import MP4
from mutagen.flac import FLAC
from mutagen.oggvorbis import OggVorbis
from mutagen.oggopus import OggOpus
import whisper_s2t
import numpy as np


class Segment(BaseModel):
    id: int
    seek: float
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float


class WhisperVerbose(BaseModel):
    task: str
    language: str
    duration: float
    text: str
    segments: List[Segment]


def get_file_duration(file_path: str) -> float:
    maudio = None
    if file_path.endswith("wav"):
        maudio = WAVE(file_path)
    elif file_path.endswith("mp3"):
        maudio = MP3(file_path)
    elif file_path.endswith("m4a"):
        maudio = MP4(file_path)
    elif file_path.endswith("flac"):
        maudio = FLAC(file_path)
    elif file_path.endswith("ogg") or file_path.endswith(
        "opus"
    ):  # Check for Ogg Vorbis/Opus or just Opus
        try:
            maudio = OggOpus(file_path)  # Try Opus first
        except Exception as e:  # If it's not Opus, try Vorbis
            maudio = OggVorbis(file_path)
    else:
        raise ValueError(f"Unsupported audio type: {file_path}")

    return maudio.info.length


def generate_verbose_json(result, file_name, lang_codes) -> WhisperVerbose:
    segments = []
    final_text = ""
    language = "unknown"
    for idx, text in enumerate(result[0]):
        segments.append(
            Segment(
                id=idx,
                seek=0,
                start=text["start"],
                end=text["end"],
                text=text["text"],
                tokens=text["tokens"],
                temperature=text["temperature"],
                avg_logprob=text["avg_logprob"],
                compression_ratio=text["compression_ratio"],
                no_speech_prob=text["no_speech_prob"],
            )
        )
        if text["language"] != "unknown" and language == "unknown":
            language = text["language"]
        final_text += text["text"] + " "

    return WhisperVerbose(
        task="transcribe",
        language=(
            (language if language != "unknown" else "en")
            if lang_codes is None
            else lang_codes
        ),
        duration=get_file_duration(file_name),
        text=final_text,
        segments=segments,
    ).model_dump()


class Predictor:
    """A Predictor class for the Whisper model"""

    def __init__(self):
        self.model_id = "large-v3-turbo"
        self.pipe = None

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        self.pipe = whisper_s2t.load_model(
            model_identifier="large-v3-turbo",
            backend="TensorRT-LLM",
            asr_options={
                "beam_size": 5,
                "best_of": 5,  # Placeholder
                "patience": 1,
                "length_penalty": 1,
                "repetition_penalty": 1.01,
                "no_repeat_ngram_size": 0,
                "compression_ratio_threshold": 2.4,  # Placeholder
                "log_prob_threshold": -1.0,  # Placeholder
                "no_speech_threshold": 0.5,  # Placeholder
                "prefix": None,  # Placeholder
                "suppress_blank": True,
                "suppress_tokens": [-1],
                "without_timestamps": True,
                "max_initial_timestamp": 1.0,
                "word_timestamps": False,  # Placeholder
                "sampling_temperature": 1,
                "return_scores": False,
                "return_no_speech_prob": False,
                "word_aligner_model": "tiny",
            },
        )

    def predict(
        self,
        audio,
        model_name="large-v3-turbo",
        transcription="verbose_json",
        translate=False,
        translation="plain_text",
        language=None,
        temperature=0,
        best_of=5,
        beam_size=5,
        patience=1,
        length_penalty=None,
        suppress_tokens="-1",
        initial_prompt=None,
        condition_on_previous_text=True,
        temperature_increment_on_fallback=0.2,
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=0.6,
        enable_vad=False,
        word_timestamps=False,
    ):
        """
        Run a single prediction on the model
        """
        pipe = self.pipe
        if not pipe:
            raise ValueError(f"Model '{model_name}' not found.")

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        # audio_x = whisperx.load_audio(audio)
        # result = pipe.transcribe(audio_x, batch_size=64)
        result = pipe.transcribe(
            [audio],
            lang_codes=[language],
            tasks=["transcribe"],
            initial_prompts=[initial_prompt],
            batch_size=12,
        )

        return generate_verbose_json(
            result=result, file_name=audio, lang_codes=language
        )

        # if transcription == "json":
        #     return result
        # elif transcription == "verbose_json":
        # return generate_verbose_json(
        #     pipeline_result=result,
        #     file_name=audio,
        #     tokenizer=pipe.model.hf_tokenizer,
        # )

        # segments, info = list(
        #     model.transcribe(
        #         str(audio),
        #         language=language,
        #         task="transcribe",
        #         beam_size=beam_size,
        #         best_of=best_of,
        #         patience=patience,
        #         length_penalty=length_penalty,
        #         temperature=temperature,
        #         compression_ratio_threshold=compression_ratio_threshold,
        #         log_prob_threshold=logprob_threshold,
        #         no_speech_threshold=no_speech_threshold,
        #         condition_on_previous_text=condition_on_previous_text,
        #         initial_prompt=initial_prompt,
        #         prefix=None,
        #         suppress_blank=True,
        #         suppress_tokens=[-1],
        #         without_timestamps=False,
        #         max_initial_timestamp=1.0,
        #         word_timestamps=word_timestamps,
        #         vad_filter=enable_vad,
        #     )
        # )

        # segments = list(segments)

        # transcription = format_segments(transcription, segments)

        # if translate:
        #     translation_segments, translation_info = model.transcribe(
        #         str(audio), task="translate", temperature=temperature
        #     )

        #     translation = format_segments(translation, translation_segments)

        # results = {
        #     "segments": serialize_segments(segments),
        #     "detected_language": info.language,
        #     "transcription": transcription,
        #     "translation": translation if translate else None,
        #     "device": "cuda" if rp_cuda.is_available() else "cpu",
        #     "model": model_name,
        # }

        # if word_timestamps:
        #     word_timestamps = []
        #     for segment in segments:
        #         for word in segment.words:
        #             word_timestamps.append(
        #                 {
        #                     "word": word.word,
        #                     "start": word.start,
        #                     "end": word.end,
        #                 }
        #             )
        #     results["word_timestamps"] = word_timestamps

        # return result


# def serialize_segments(transcript):
#     """
#     Serialize the segments to be returned in the API response.
#     """
#     return [
#         {
#             "id": segment.id,
#             "seek": segment.seek,
#             "start": segment.start,
#             "end": segment.end,
#             "text": segment.text,
#             "tokens": segment.tokens,
#             "temperature": segment.temperature,
#             "avg_logprob": segment.avg_logprob,
#             "compression_ratio": segment.compression_ratio,
#             "no_speech_prob": segment.no_speech_prob,
#         }
#         for segment in transcript
#     ]


# def format_segments(format, segments):
#     """
#     Format the segments to the desired format
#     """

#     if format == "plain_text":
#         return " ".join([segment.text.lstrip() for segment in segments])
#     elif format == "formatted_text":
#         return "\n".join([segment.text.lstrip() for segment in segments])
#     elif format == "srt":
#         return write_srt(segments)

#     return write_vtt(segments)


# def write_vtt(transcript):
#     """
#     Write the transcript in VTT format.
#     """
#     result = ""

#     for segment in transcript:
#         result += (
#             f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}\n"
#         )
#         result += f"{segment.text.strip().replace('-->', '->')}\n"
#         result += "\n"

#     return result


# def write_srt(transcript):
#     """
#     Write the transcript in SRT format.
#     """
#     result = ""

#     for i, segment in enumerate(transcript, start=1):
#         result += f"{i}\n"
#         result += f"{format_timestamp(segment.start, always_include_hours=True, decimal_marker=',')} --> "
#         result += f"{format_timestamp(segment.end, always_include_hours=True, decimal_marker=',')}\n"
#         result += f"{segment.text.strip().replace('-->', '->')}\n"
#         result += "\n"

#     return result
