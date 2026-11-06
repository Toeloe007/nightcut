"""Ask a local vision model what happens in each candidate clip."""

from __future__ import annotations

import json
from pathlib import Path

from . import llm
from .media import MediaInfo, grab_frames

SYSTEM = """You review gameplay footage for a YouTube editor.
You get several frames from one short moment, in chronological order, and what the player said.
Answer only with JSON:
{"event": one of {events},
 "intensity": integer 0-10 (how entertaining this moment is to watch),
 "summary": one short sentence in German describing what happens}
"success": goal reached, level up, record, big sale, win. "fail": mistake, chaos, something breaks.
"funny": a funny situation. "reaction": ordinary picture, but the player reacts strongly or tells something entertaining.
Be strict: menus, walking, waiting, loading screens and the map without a reaction are "nothing" with intensity 0-2."""

SHOOTER_EVENTS = ("multikill", "kill", "vehicle_kill", "explosion", "clutch", "death")


def is_shooter(cfg: dict | None) -> bool:
    """Shooter events are only offered for shooters: elsewhere the model calls a busy kitchen a 'multikill'."""
    profile = (cfg or {}).get("profile") or {}
    return profile.get("game_type", "shooter" if profile.get("killfeed_region") else "other") == "shooter"


def offered_events(cfg: dict) -> list[str]:
    """The events the vision model may answer with for this game."""
    shooter = is_shooter(cfg)
    return [e for e in event_weights(cfg) if shooter or e not in SHOOTER_EVENTS]


def frame_times(start: float, end: float, peak: float, count: int) -> list[float]:
    if count <= 1:
        return [peak]
    step = (end - start) / (count + 1)
    times = sorted({round(start + step * (i + 1), 2) for i in range(count - 1)} | {peak})
    return times[:count]


def analyse(info: MediaInfo, candidates: list[dict], cfg: dict, progress=None,
            endpoints: list[dict] | None = None, per_worker: list[int] | None = None) -> list[dict]:
    """endpoints: one Ollama config per GPU; all GPUs take candidates from one queue."""
    from .gpus import parallel_map

    vcfg = cfg["vision"]
    endpoints = endpoints or [cfg["ollama"]]
    hints = cfg["profile"].get("vision_hints", "")
    system = SYSTEM.replace("{events}", json.dumps(offered_events(cfg)))
    todo = [c for c in candidates if not c.get("vision")]  # some are already known, e.g. from the kill feed

    def look(cand: dict, n: int) -> None:
        times = frame_times(cand["start"], cand["end"], cand["peak"], vcfg["frames_per_candidate"])
        frames = grab_frames(Path(info.path), times, cfg["layout"]["gameplay"], info.width, info.height, vcfg["frame_width"])
        user = f"{hints}\nFrames at seconds {times}. The player said: \"{cand['transcript'][:400]}\""
        try:
            result = llm.chat_json(endpoints[n], vcfg["model"], system, user, images=frames, keep_loaded=True)
        except llm.OllamaError:
            raise
        except Exception as exc:  # one bad answer should not stop the night
            result = {"event": "unknown", "intensity": 5, "summary": f"vision failed: {exc}"}
        cand["vision"] = {
            "event": str(result.get("event", "unknown")),
            "intensity": int(result.get("intensity", 5) or 0),
            "summary": str(result.get("summary", "")),
        }

    try:
        parallel_map(todo, list(range(len(endpoints))), look, progress=progress, per_worker=per_worker)
    finally:
        for ocfg in endpoints:
            llm.unload(ocfg, vcfg["model"])
    return candidates


# How much a type of moment is worth in a highlight video. A death alone is rarely
# a highlight; the audio/transcript part of the score still lifts funny reactions.
EVENT_WEIGHT = {"multikill": 1.0, "clutch": 1.0, "vehicle_kill": 0.9, "funny": 0.9, "kill": 0.75,
                "success": 0.8, "fail": 0.8, "reaction": 0.7,
                "explosion": 0.6, "death": 0.3, "nothing": 0.0, "unknown": 0.5}


def event_weights(cfg: dict | None = None) -> dict[str, float]:
    """Built-in event weights plus game-specific ones from the profile (vision_events: {name: weight})."""
    extra = ((cfg or {}).get("profile") or {}).get("vision_events") or {}
    weights = {k: v for k, v in EVENT_WEIGHT.items() if k != "unknown"}
    weights.update({str(k): float(v) for k, v in extra.items()})
    return weights


def rescore(candidates: list[dict], cfg: dict | None = None) -> None:
    """Blend audio/transcript score with the vision rating into one 0..1 value."""
    weights = {**event_weights(cfg), "unknown": EVENT_WEIGHT["unknown"]}
    top = max((c["score"] for c in candidates), default=1.0) or 1.0
    for c in candidates:
        audio = c["score"] / top
        vis = c.get("vision")
        if not vis:
            c["final_score"] = round(audio, 4)
            continue
        kills = c.get("kills", 0)
        if kills and weights.get(vis["event"], 0.5) < weights["kill"]:
            # the kill feed is more reliable than a few frames: the moment does contain kills
            vis.update(event="multikill" if kills > 1 else "kill", intensity=max(vis["intensity"], min(10, 5 + 2 * kills)))
        visual = vis["intensity"] / 10 * weights.get(vis["event"], 0.5)
        # In dialogue-led GTA-RP, static gameplay must not suppress strong
        # spoken moments. The current audio score is only a proxy for dialogue
        # quality; actual story relevance is judged by story_editor from text.
        story_cfg = (cfg or {}).get('story_mode') or {}
        dialogue_led = bool(story_cfg.get('enabled'))
        if dialogue_led:
            audio_weight = float(story_cfg.get('audio_weight', 0.75))
            audio_weight = max(0.0, min(1.0, audio_weight))
        else:
            audio_weight = 0.45
        c["final_score"] = round(audio_weight * audio + (1 - audio_weight) * visual, 4)


STORY_START_SYSTEM = """Du untersuchst chronologisch geordnete GTA-RP-Videoframes aus einem
kurzen Zeitfenster. Gesucht ist der ANFANG eines konkreten Vorfalls, nicht irgendeine
Autofahrt. Einzelbilder beweisen keinen Fluchtbeginn; verwende sichtbare Hinweise
und zeitgestempelten Dialog zusammen. WICHTIG: Wenn die Bilder nur ein
Polizeiauto auf einer Straße zeigen, ist das KEIN Widerspruch zu einem
klaren Funkspruch über den Beginn einer Verfolgung. Bewerte nur die
Sichtbarkeit; die Dialogentscheidung trifft eine separate Instanz. Antworte ausschließlich JSON:
{"start_visible":true/false,"confidence":"high|medium|low",
 "evidence":"konkreter Bild-/Dialoghinweis oder warum unklar",
 "suggested_s":null}
start_visible=true nur wenn ein tatsächlicher Übergang von normaler Situation zu
Verfolgung/Flucht erkennbar ist. Bei bereits laufender Verfolgung false.
Wenn keine belastbare Zeit sichtbar ist, suggested_s=null."""


def verify_story_start(info: MediaInfo, cfg: dict, windows: list[dict]) -> list[dict]:
    """Inspect only short-listed windows, never reprocess the whole recording."""
    results = []
    vcfg = cfg['vision']
    for w in windows[:8]:
        a, b = max(0.0, float(w['start'])), min(float(info.duration), float(w['end']))
        if b - a < 3:
            continue
        times = [round(a + (b - a) * fraction, 2) for fraction in (.1, .3, .5, .7, .9)]
        try:
            frames = grab_frames(Path(info.path), times, cfg['layout']['gameplay'],
                                 info.width, info.height, vcfg['frame_width'])
            prompt = ('Story: ' + str(w.get('story', '')) + '\n'
                      + 'Frames bei Sekunden ' + json.dumps(times) + '\n'
                      + 'Dialog: ' + str(w.get('dialogue', ''))[:1100])
            response = llm.chat_json(cfg['ollama'], vcfg['model'], STORY_START_SYSTEM,
                                     prompt, images=frames, keep_loaded=True)
            results.append({**w, 'frame_times': times, 'vision': response})
        except Exception as exc:
            results.append({**w, 'frame_times': times, 'error': repr(exc)})
    if results:
        llm.unload(cfg['ollama'], vcfg['model'])
    return results
