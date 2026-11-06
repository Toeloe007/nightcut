"""Render vertical Shorts: facecam on top, gameplay below, word-by-word captions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .media import AudioStream, MediaInfo, crop_filter, encoder_gpu, run

W, H = 1080, 1920


def pick(clips: list[dict], count: int, min_s: float) -> list[dict]:
    """Strongest clips of the edit, best first."""
    usable = [c for c in clips if c["end"] - c["start"] >= min_s]
    return sorted(usable, key=lambda c: c.get("final_score", c["score"]), reverse=True)[:count]


def word_timestamps(pcm: np.ndarray, sample_rate: int, start: float, end: float, cfg: dict, asr) -> list[dict]:
    audio = np.asarray(pcm[int(start * sample_rate):int(end * sample_rate)], dtype=np.float32) / 32768.0
    res = asr({"raw": audio, "sampling_rate": sample_rate}, return_timestamps="word",
              generate_kwargs={"language": cfg["transcribe"]["language"], "task": "transcribe"})
    words = []
    for ch in res.get("chunks", []):
        a, b = ch["timestamp"]
        text = ch["text"].strip()
        if text and a is not None:
            words.append({"start": a, "end": b if b is not None else a + 0.4, "text": text})
    return words


def _ass_time(t: float) -> str:
    t = max(t, 0.0)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def _clean(word: str) -> str:
    return word.strip(" ,.;:").upper()


def _ass_color(hex_rgb: str) -> str:
    r, g, b = hex_rgb.lstrip("#")[0:2], hex_rgb.lstrip("#")[2:4], hex_rgb.lstrip("#")[4:6]
    return f"&H00{b}{g}{r}".upper()


def write_ass(words: list[dict], path: Path, scfg: dict, caption_y: int) -> None:
    """Groups of a few words; the word being spoken is highlighted."""
    hi = _ass_color(scfg["highlight_color"])
    lines = [
        "[Script Info]", "ScriptType: v4.00+", f"PlayResX: {W}", f"PlayResY: {H}", "WrapStyle: 2", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
        "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, "
        "MarginR, MarginV, Encoding",
        f"Style: Cap,{scfg['font']},{scfg['font_size']},&H00FFFFFF,&H00FFFFFF,&H00000000,&H64000000,-1,0,0,0,"
        f"100,100,2,0,1,{scfg['outline']},3,5,60,60,0,1",
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    n = scfg["words_per_caption"]
    for g in range(0, len(words), n):
        group = words[g:g + n]
        for i, w in enumerate(group):
            # Whisper sometimes stretches a word back into the silence before it.
            start = max(w["start"], w["end"] - 0.8)
            end = group[i + 1]["start"] if i + 1 < len(group) else w["end"] + 0.25
            end = min(end, w["end"] + 0.6)  # do not keep text on screen through a pause
            text = " ".join(
                (f"{{\\c{hi}}}{_clean(x['text'])}{{\\c&H00FFFFFF}}" if j == i else _clean(x["text"]))
                for j, x in enumerate(group)
            )
            lines.append(f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Cap,,0,0,0,,"
                         f"{{\\pos({W // 2},{caption_y})}}{text}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _filter_path(p: Path) -> str:
    """Escape a Windows path for use inside an ffmpeg filter argument."""
    return p.resolve().as_posix().replace(":", r"\:").replace("'", r"\'")


def render(info: MediaInfo, clip: dict, n: int, streams: dict[str, AudioStream], ass: Path, cfg: dict,
           out_dir: Path, gpu: int | None = None) -> Path:
    scfg = cfg["shorts"]
    out = out_dir / f"short_{n:02d}.mp4"
    start = clip["start"]
    dur = min(clip["end"] - clip["start"], scfg["max_s"])
    lay = cfg["layout"]

    # A portrait VOD with a burned-in facecam already has its final layout.
    # Preserve the full image instead of cropping and stacking it again.
    if info.width * 16 == info.height * 9 and lay.get("facecam_burned_in", False):
        graph = f"[0:v]scale={W}:{H},setsar=1[stack];"
        src = "[stack]"
    elif lay.get("facecam"):
        cam_h = round(W * 9 / 16 / 2) * 2                      # 16:9 camera across the full width
        game_h = H - cam_h
        graph = (f"[0:v]{crop_filter(lay['facecam'], info.width, info.height)},scale={W}:{cam_h}:force_original_aspect_ratio=increase,crop={W}:{cam_h}[cam];"
                 f"[0:v]{crop_filter(lay['gameplay'], info.width, info.height)},"
                 f"crop=ih*{W}/{game_h}:ih,scale={W}:{game_h}[game];[cam][game]vstack[stack];")
        src = "[stack]"
    else:
        graph = f"[0:v]{crop_filter(lay['gameplay'], info.width, info.height)},crop=ih*9/16:ih,scale={W}:{H}[stack];"
        src = "[stack]"
    graph += f"{src}null[v];"
    audio_in = "".join(f"[0:{s.index}]" for s in streams.values())
    graph += f"{audio_in}amix=inputs={len(streams)}:normalize=0:dropout_transition=0[a]"

    run(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-ss", f"{start:.3f}", "-t", f"{dur:.3f}",
         "-i", str(info.path), "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
         "-c:v", cfg["render"]["codec"], *encoder_gpu(cfg["render"]["codec"], gpu), "-preset", "p5", "-rc", "vbr", "-cq", "20", "-b:v", "0",
         "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", str(out)])
    return out


def caption_y(cfg: dict) -> int:
    """Captions sit on the edge between facecam and gameplay."""
    if cfg["layout"].get("facecam"):
        return round(W * 9 / 16 / 2) * 2 + 40
    return int(H * 0.7)
