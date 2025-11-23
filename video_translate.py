"""
Video Translator Prototype (Streamlit-ready)
Author: ChatGPT (prototype, updated)

This file contains:
- Core pipeline functions (audio extraction, transcription via Whisper, translation, TTS, subtitle creation, and video muxing)
- Robust CLI-safe argument parsing (for developers who want to run from terminal)
- A Streamlit web UI so you can run this tool as a web app (Option 3). The Streamlit app is included at the bottom of the file.

Notes & limitations:
- External dependencies required for full functionality: ffmpeg (system), whisper (Python package), deep-translator, gTTS, pydub, and streamlit.
- The Streamlit app runs the same pipeline synchronously; long files (movies) will take significant time and may require running on a machine with sufficient CPU/RAM. For production, consider converting processing to background jobs (Celery/RQ) or using cloud workers.
- Respect copyright: only process videos you have rights to.

USAGE (Streamlit):
1) Install dependencies (see README in comments).
2) Run:
   streamlit run video_translate.py
3) Open the URL shown by Streamlit and use the web UI to upload a video and choose languages/mode.

If you prefer CLI, the usual example still works:
python video_translate.py --input in.mp4 --source_lang en --target_lang hi --out out_translated.mp4 --mode subtitles
"""

import argparse
import os
import subprocess
import tempfile
import math
import sys
from pathlib import Path
from typing import List, Optional

# Optional Streamlit import (only required for the web UI)
try:
    import streamlit as st
except Exception:
    st = None

# Lazy imports: keep original try/excepts so missing optional deps don't break simple runs/tests
try:
    import whisper
except Exception:
    whisper = None

try:
    from deep_translator import GoogleTranslator
except Exception:
    GoogleTranslator = None

try:
    from gtts import gTTS
except Exception:
    gTTS = None

try:
    import pysrt
except Exception:
    pysrt = None

try:
    from pydub import AudioSegment
except Exception:
    AudioSegment = None


def run_cmd(cmd: List[str]):
    print("RUN:", " ".join(cmd))
    subprocess.check_call(cmd)


def extract_audio(video_path: str, out_audio: str, sample_rate: int = 16000):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "wav",
        out_audio,
    ]
    run_cmd(cmd)


def transcribe_with_whisper(audio_path: str, model_name: str = "small") -> List[dict]:
    if whisper is None:
        raise RuntimeError(
            "whisper package not installed. Install via: pip install git+https://github.com/openai/whisper.git"
        )
    print("Loading whisper model:", model_name)
    model = whisper.load_model(model_name)
    print("Transcribing (this may take time)")
    result = model.transcribe(audio_path)
    segments = result.get("segments", [])
    return segments


def translate_segments(
    segments: List[dict], source_lang: str, target_lang: str
) -> List[dict]:
    if GoogleTranslator is None:
        raise RuntimeError("deep_translator not installed. pip install deep-translator")
    out: List[dict] = []
    translator = GoogleTranslator(source=source_lang, target=target_lang)
    for seg in segments:
        text = seg.get("text", "").strip()
        translated = "" if not text else translator.translate(text)
        out.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": text,
                "translated": translated,
            }
        )
    return out


def segments_to_srt(segments: List[dict], srt_path: str):
    def fmt_time(t: float) -> str:
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - math.floor(t)) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, start=1):
            f.write(str(i) + "\n")
            f.write(f"{fmt_time(seg['start'])} --> {fmt_time(seg['end'])}\n")
            text = seg.get("translated", "")
            f.write(text.replace("\n", " ").strip() + "\n\n")


def tts_segments_to_audio(
    segments: List[dict], lang_code: str, out_audio_path: str
):
    if gTTS is None or AudioSegment is None:
        raise RuntimeError("gTTS and pydub required. pip install gTTS pydub")

    parts: List[AudioSegment] = []
    for seg in segments:
        text = seg.get("translated", "").strip()
        if not text:
            duration_ms = int((seg["end"] - seg["start"]) * 1000)
            parts.append(AudioSegment.silent(duration=duration_ms))
            continue

        tts = gTTS(text=text, lang=lang_code)
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp.close()
        tts.save(tmp.name)

        part = AudioSegment.from_file(tmp.name, format="mp3")
        seg_duration_ms = int((seg["end"] - seg["start"]) * 1000)
        if len(part) < seg_duration_ms:
            part = part + AudioSegment.silent(
                duration=(seg_duration_ms - len(part))
            )
        elif len(part) > seg_duration_ms + 300:
            part = part[:seg_duration_ms]
        parts.append(part)
        os.unlink(tmp.name)

    output = AudioSegment.silent(duration=0)
    for p in parts:
        output += p
    output.export(out_audio_path, format="wav")


def replace_audio_in_video(original_video: str, new_audio: str, out_video: str):
    if not os.path.exists(original_video):
        raise FileNotFoundError(f"Original video not found: {original_video}")
    if not os.path.exists(new_audio):
        raise FileNotFoundError(f"Replacement audio not found: {new_audio}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        original_video,
        "-i",
        new_audio,
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-c:v",
        "copy",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        out_video,
    ]
    run_cmd(cmd)


def burn_subtitles_into_video(video_in: str, srt_path: str, video_out: str):
    if not os.path.exists(video_in):
        raise FileNotFoundError(f"Input video not found: {video_in}")
    if not os.path.exists(srt_path):
        raise FileNotFoundError(f"Subtitle file not found: {srt_path}")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        video_in,
        "-vf",
        f"subtitles={srt_path}",
        "-c:a",
        "copy",
        video_out,
    ]
    run_cmd(cmd)


def chunk_segments_for_memory(
    segments: List[dict], max_chunk_sec: int = 600
) -> List[List[dict]]:
    chunks: List[List[dict]] = []
    cur: List[dict] = []
    cur_start: Optional[float] = None
    for seg in segments:
        if cur_start is None:
            cur_start = seg["start"]
        if seg["end"] - cur_start > max_chunk_sec and cur:
            chunks.append(cur)
            cur = [seg]
            cur_start = seg["start"]
        else:
            cur.append(seg)
    if cur:
        chunks.append(cur)
    return chunks


# ---------- Argument parsing helpers ----------
def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Video Translator - extract audio, transcribe, translate and optionally dub or burn subtitles."
    )
    parser.add_argument("--input", required=True, help="input video path")
    parser.add_argument(
        "--source_lang",
        default="en",
        help="source language code for translation (e.g. en)",
    )
    parser.add_argument(
        "--target_lang",
        default="hi",
        help="target language code for translation/gTTS (e.g. hi)",
    )
    parser.add_argument(
        "--out",
        default="out_translated.mp4",
        help="output video path",
    )
    parser.add_argument(
        "--mode",
        choices=["subtitles", "dubbed"],
        default="subtitles",
        help="subtitles or dubbed",
    )
    parser.add_argument(
        "--whisper_model",
        default="small",
        help="whisper model name (tiny, base, small, medium, large)",
    )
    return parser


def get_args(argv: Optional[List[str]] = None) -> Optional[argparse.Namespace]:
    """Parse args safely. If parsing fails (missing required args), print helpful example and return None.

    argv: supply a list of args (excluding program name) if calling programmatically.
    """
    parser = get_parser()
    try:
        args = parser.parse_args(argv)
        return args
    except SystemExit:
        print("\nIt looks like required arguments were missing or invalid.\n")
        example = (
            "python video_translate.py --input in.mp4 --source_lang en "
            "--target_lang hi --out out_translated.mp4 --mode subtitles"
        )
        print("Example usage:")
        print("  " + example + "\n")
        return None


def validate_args(args: argparse.Namespace) -> bool:
    if args is None:
        return False
    if not os.path.exists(args.input):
        print(f"Error: input file does not exist: {args.input}")
        return False
    return True


def run_pipeline(args: argparse.Namespace):
    """Run the main pipeline. This function is separated from CLI handling for easier testing."""
    video_in = args.input
    workdir = tempfile.mkdtemp(prefix="vidtrans_")
    audio_wav = os.path.join(workdir, "audio.wav")

    print("Extracting audio...")
    extract_audio(video_in, audio_wav)

    print("Transcribing...")
    segments = transcribe_with_whisper(audio_wav, model_name=args.whisper_model)

    print("Translating...")
    translated = translate_segments(
        segments, source_lang=args.source_lang, target_lang=args.target_lang
    )

    srt_path = os.path.join(workdir, "translated.srt")
    segments_to_srt(translated, srt_path)
    print("Saved translated subtitles to", srt_path)

    if args.mode == "subtitles":
        print("Burning subtitles into video (this may be slow)")
        burn_subtitles_into_video(video_in, srt_path, args.out)
        print("Output saved to", args.out)
    else:
        print("Generating TTS audio (may be slow)")
        tts_audio = os.path.join(workdir, "tts.wav")
        tts_segments_to_audio(
            translated, lang_code=args.target_lang, out_audio_path=tts_audio
        )
        out_with_audio = os.path.join(workdir, "video_dubbed.mp4")
        replace_audio_in_video(video_in, tts_audio, out_with_audio)
        burn_subtitles_into_video(out_with_audio, srt_path, args.out)
        print("Dubbed output saved to", args.out)


# ---------- Simple unit tests (do not require heavy deps) ----------
def _test_arg_parsing_ok():
    argv = [
        "--input",
        "movie.mp4",
        "--source_lang",
        "en",
        "--target_lang",
        "hi",
        "--mode",
        "subtitles",
    ]
    args = get_args(argv)
    assert args is not None
    assert args.input == "movie.mp4"
    assert args.mode == "subtitles"
    print("TEST: arg_parsing_ok PASSED")


def _test_arg_parsing_missing():
    args = get_args([])
    assert args is None
    print("TEST: arg_parsing_missing PASSED")


def _test_validate_args_nonexistent_file():
    class DummyArgs:
        input = "this_file_does_not_exist_12345.mp4"

    ok = validate_args(DummyArgs)
    assert ok is False
    print("TEST: validate_args_nonexistent_file PASSED")


def run_tests():
    print("Running lightweight unit tests...")
    _test_arg_parsing_ok()
    _test_arg_parsing_missing()
    _test_validate_args_nonexistent_file()
    print("All tests passed.")


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = get_args(argv)
    if args is None:
        return 0

    if not validate_args(args):
        print("\nValidation failed. Please fix the issues above and try again.")
        return 1

    try:
        run_pipeline(args)
    except Exception as e:
        print("\nPipeline error:", str(e))
        print(
            "Make sure external dependencies (ffmpeg) and optional Python packages are installed."
        )
        return 2
    return 0


# ---------- Streamlit web UI (Option 3) ----------

def run_streamlit_app():
    if st is None:
        raise RuntimeError("streamlit not installed. Run: pip install streamlit")

    # Import YouTube downloader inside the function so CLI usage does not require it
    try:
        import yt_dlp
    except Exception:
        yt_dlp = None

    # Basic page config
    st.set_page_config(page_title="Rayvia AI Video Translator", layout="wide")

    st.markdown(
        """
    <h2 style="text-align:center; font-size:32px; font-weight:700; margin-bottom:4px;">
        Rayvia AI – Video Translator & Dubbing Tool
    </h2>
    <p style="text-align:center; font-size:14px; color:#4b5563;">
        Upload a video or paste a YouTube link → Choose languages → Translate → Download
    </p>
    """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([1, 1])

    video_path: Optional[str] = None

    with col1:
        st.subheader("📤 Upload Video File")
        uploaded = st.file_uploader(
            "Upload video file",
            type=["mp4", "mov", "mkv", "avi"],
            help="Max size depends on server limits.",
        )
        if uploaded is not None:
            tmpdir_up = tempfile.mkdtemp(prefix="upload_")
            video_path = os.path.join(tmpdir_up, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Uploaded: {uploaded.name}")

    with col2:
        st.subheader("🔗 Paste YouTube Video Link")
        youtube_url = st.text_input(
            "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
        )
        if youtube_url:
            if yt_dlp is None:
                st.error(
                    "yt_dlp is not installed on the server. Ask the admin to install: pip install yt-dlp"
                )
            else:
                if st.button("Download from YouTube"):
                    try:
                        st.info("Downloading video from YouTube. Please wait...")
                        tmpdir_yt = tempfile.mkdtemp(prefix="yt_")
                        ydl_opts = {
                            "outtmpl": os.path.join(tmpdir_yt, "video.%(ext)s"),
                            "format": "bestvideo+bestaudio/best",
                        }
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            ydl.download([youtube_url])

                        for fname in os.listdir(tmpdir_yt):
                            if fname.lower().endswith((".mp4", ".mkv", ".mov", ".avi")):
                                video_path = os.path.join(tmpdir_yt, fname)
                                break

                        if video_path:
                            st.success("YouTube video downloaded successfully!")
                        else:
                            st.error("Could not find downloaded video file.")
                    except Exception as e:
                        st.error(f"Error downloading video: {e}")

    st.markdown("---")

    if not video_path:
        st.warning("Please upload a video OR download from a YouTube link.")
        return

    st.markdown("### 📝 Translation Settings")

    colA, colB = st.columns(2)

    with colA:
        src_lang = st.selectbox(
            "Source Language",
            ["auto", "en", "hi", "ur", "ar", "fr", "es", "ru", "zh", "ja"],
            index=1,
        )
        tgt_lang = st.selectbox(
            "Target Language",
            ["hi", "en", "ur", "ar", "fr", "es", "ru", "zh", "ja"],
            index=0,
        )

    with colB:
        mode = st.selectbox("Mode", ["subtitles", "dubbed"], index=0)
        whisper_model = st.selectbox(
            "Whisper Model", ["tiny", "base", "small", "medium", "large"], index=2
        )

    st.markdown("---")

    if st.button("🚀 Start Translation"):
        with st.spinner(
            "Processing video… This can take a long time for bigger files. Please wait..."
        ):
            out_path = os.path.join(
                tempfile.gettempdir(), f"rayvia_output_{tgt_lang}.mp4"
            )
            args = argparse.Namespace(
                input=video_path,
                source_lang=src_lang,
                target_lang=tgt_lang,
                out=out_path,
                mode=mode,
                whisper_model=whisper_model,
            )
            try:
                if not validate_args(args):
                    st.error(
                        "Validation failed. Check that the input file exists and try again."
                    )
                else:
                    run_pipeline(args)
                    st.success("Translation complete!")

                    if os.path.exists(args.out):
                        with open(args.out, "rb") as f:
                            st.download_button(
                                "⬇ Download Translated Video",
                                data=f,
                                file_name=os.path.basename(args.out),
                            )
                    else:
                        st.warning(
                            "Output file not found. Check server logs if the problem continues."
                        )
            except Exception as e:
                st.error(f"Pipeline error: {e}")


if __name__ == "__main__":
    if st is not None:
        run_streamlit_app()
    else:
        if len(sys.argv) <= 1:
            print(
                "No CLI arguments provided. Running built-in lightweight tests and showing example usage."
            )
            run_tests()
            print("\nExample command:")
            print(
                "  python video_translate.py --input in.mp4 --source_lang en --target_lang hi --out out_translated.mp4 --mode subtitles"
            )
            sys.exit(0)

        exit_code = main()
        sys.exit(exit_code)
