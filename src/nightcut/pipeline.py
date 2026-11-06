"""The overnight pipeline: resumable stages, each writing its result to the work folder.

If the night is interrupted, run the same command again: finished stages are skipped.
"""

from __future__ import annotations

import contextlib
import gc
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from rich.console import Console

from . import gpus  # first: sets CUDA_DEVICE_ORDER before PyTorch is loaded
from . import context_shorts, story_editor, costs, editor, export_xmeml, killfeed, metadata, shorts, signals, summary, thumbnails, vision
from .media import extract_audio_tracks, load_pcm, probe
from .render import render_clip
from .transcribe import free_vram, load_asr, transcribe_regions

console = Console()


class VramMonitor:
    """Sample used GPU memory once per second, per GPU.

    Windows does not report per-process VRAM, so the first sample is taken as
    baseline (desktop, browser, other apps) and only the increase is reported.
    Energy is the sum over all GPUs in the PC.
    """

    def __init__(self):
        self.baselines: list[int] | None = None
        self.peaks: list[int] = []
        self.energy_wh = 0.0     # GPU energy, integrated from power.draw
        self.seconds = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,power.draw",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5).stdout
                rows = [line.split(",") for line in out.strip().splitlines()]
                used = [int(r[0]) for r in rows]
                self.energy_wh += sum(float(r[1]) for r in rows) / 3600  # one sample per second
                if self.baselines is None:
                    self.baselines, self.peaks = list(used), list(used)
                self.peaks = [max(p, u) for p, u in zip(self.peaks, used)]
            except Exception:
                pass
            self.seconds += 1.0
            self._stop.wait(1.0)

    @property
    def used_by_gpu(self) -> list[int]:
        return [max(p - b, 0) for p, b in zip(self.peaks, self.baselines or [])]

    @property
    def used_mib(self) -> int:
        return max(self.used_by_gpu, default=0)

    @property
    def baseline_mib(self) -> int | None:
        return max(self.baselines) if self.baselines else None

    def __enter__(self):
        self._thread.start()
        time.sleep(1.2)  # take the baseline before the stage loads anything
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=3)


def _time_text(stage: dict) -> str:
    text = f"{stage['seconds'] / 60:.1f} min"
    if stage.get("parallel"):
        text += f" (at the same time as {stage['parallel']})"
    return text


def _vram_text(stage: dict) -> str:
    """'11.2 GB', or per GPU '9.8 / 6.1 GB' for stages measured on several cards."""
    per_gpu = stage.get("vram_by_gpu_mib") or []
    if len(per_gpu) > 1:
        return " / ".join(f"{m / 1024:.1f}" for m in per_gpu) + " GB"
    return f"{stage['vram_peak_mib'] / 1024:.1f} GB"


class Run:
    def __init__(self, source: Path, cfg: dict, force: set[str] | None = None, name: str | None = None,
                 use_gpus: str | None = None):
        self.source = source
        self.cfg = cfg
        self.use_gpus = use_gpus  # "auto" or a number of GPUs (overrides gpus.use in the config)
        self.gpu_list: list[dict] = []
        self.endpoints: list[dict] = [cfg["ollama"]]
        self._split: dict[str, list[int]] = {"killfeed": [], "vision": []}  # model calls per GPU
        self.force = force or set()
        self._dirty = False  # once a stage re-runs, everything after it must re-run too
        self.work = Path(os.path.abspath(cfg["output_dir"])) / (name or source.stem)
        self.work.mkdir(parents=True, exist_ok=True)
        self.log_path = self.work / "run.json"
        self.log = json.loads(self.log_path.read_text(encoding="utf-8")) if self.log_path.exists() else {"stages": {}}
        self.pricing = costs.load_pricing(Path(cfg["pricing_file"]) if cfg.get("pricing_file") else None)

    # -- helpers -------------------------------------------------------------
    def _json(self, name: str):
        return self.work / f"{name}.json"

    def _load(self, name: str):
        return json.loads(self._json(name).read_text(encoding="utf-8"))

    def _save(self, name: str, data) -> None:
        self._json(name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _stage(self, name: str, fn, *, cached_file: str | None = None):
        """Run fn unless its result already exists; record wall time and VRAM peak."""
        target = cached_file or name
        if self._json(target).exists() and name not in self.force and not self._dirty:
            console.print(f"[dim]✓ {name} (cached)[/dim]")
            return self._load(target)
        self._dirty = True
        console.print(f"[bold cyan]▶ {name}[/bold cyan]")
        costs.usage.reset()
        t0 = time.time()
        with VramMonitor() as vram:
            result = fn()
        elapsed = time.time() - t0
        self._save(target, result)
        overhead_wh = self.pricing["electricity"]["system_overhead_w"] * elapsed / 3600
        self.log["stages"][name] = {"seconds": round(elapsed, 1), "vram_peak_mib": vram.used_mib,
                                    "vram_by_gpu_mib": vram.used_by_gpu,
                                    "vram_baseline_mib": vram.baseline_mib,
                                    "energy_wh": round(vram.energy_wh + overhead_wh, 2),
                                    "usage": costs.usage.as_dict(),
                                    **({"by_gpu": self._gpu_shares(name)} if self._gpu_shares(name) else {})}
        self.log_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        console.print(f"[green]✓ {name}[/green] {elapsed / 60:.1f} min, VRAM used {vram.used_mib / 1024:.1f} GB")
        return result

    def _together(self, specs: list[tuple[str, object]]) -> dict:
        """Run several stages at the same time (on different GPUs).

        Time, energy and usage are measured once for the group and booked on the stage that
        took longest; the others are marked `parallel`, so nothing is counted twice.
        """
        results, todo = {}, []
        for name, fn in specs:
            if self._json(name).exists() and name not in self.force and not self._dirty:
                console.print(f"[dim]✓ {name} (cached)[/dim]")
                results[name] = self._load(name)
            else:
                todo.append((name, fn))
        if not todo:
            return results
        if len(todo) == 1:
            name, fn = todo[0]
            results[name] = self._stage(name, fn)
            return results
        self._dirty = True
        console.print(f"[bold cyan]▶ {' + '.join(n for n, _ in todo)}[/bold cyan] [dim](at the same time)[/dim]")
        costs.usage.reset()
        took: dict[str, float] = {}
        errors: list[BaseException] = []
        t0 = time.time()

        def work(name, fn):
            try:
                results[name] = fn()
            except BaseException as exc:  # noqa: BLE001 - re-raised below
                errors.append(exc)
            took[name] = time.time() - t0

        with VramMonitor() as vram:
            threads = [threading.Thread(target=work, args=spec, daemon=True) for spec in todo]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        if errors:
            raise errors[0]
        wall = time.time() - t0
        main = max(took, key=took.get)
        overhead_wh = self.pricing["electricity"]["system_overhead_w"] * wall / 3600
        for name, _ in todo:
            self._save(name, results[name])
            entry = {"seconds": round(took[name], 1), "vram_peak_mib": vram.used_mib,
                     "vram_by_gpu_mib": vram.used_by_gpu, "vram_baseline_mib": vram.baseline_mib,
                     **({"by_gpu": self._gpu_shares(name)} if self._gpu_shares(name) else {})}
            if name == main:
                entry.update(seconds=round(wall, 1), energy_wh=round(vram.energy_wh + overhead_wh, 2),
                             usage=costs.usage.as_dict())
            else:
                entry.update(energy_wh=0.0, usage={}, parallel=main)  # time and energy are inside `main`
            self.log["stages"][name] = entry
            console.print(f"[green]✓ {name}[/green] {took[name] / 60:.1f} min")
        self.log_path.write_text(json.dumps(self.log, indent=2), encoding="utf-8")
        console.print(f"  together {wall / 60:.1f} min, VRAM used "
                      + " / ".join(f"{m / 1024:.1f} GB" for m in vram.used_by_gpu))
        return results

    # -- GPUs -----------------------------------------------------------------
    @contextlib.contextmanager
    def _gpus(self):
        """One GPU: use the Ollama server that is already running. Several GPUs in the PC: start one
        Ollama server per GPU that nightcut uses (also with --gpus 1, so the model stays on the main card)."""
        available = gpus.list_gpus()
        gcfg = self.cfg.get("gpus") or {}
        big = [g for g in available if g["memory_mib"] >= float(gcfg.get("min_vram_gb", 10)) * 1024] or available[:1]
        self.gpu_list = gpus.choose(big, gcfg, self.use_gpus)
        # pin every step to a GPU when the PC has several (devices: [0, 0] runs two servers on one card, for testing)
        self._several_in_pc = len(available) >= 2 or len(self.gpu_list) >= 2
        if not self._several_in_pc or not self.gpu_list:
            yield
            return
        names = " + ".join(f"{g['name']} (GPU {g['index']})" for g in self.gpu_list)
        console.print(f"GPUs: {names}" + (" [dim]→ sharing the work[/dim]" if len(self.gpu_list) > 1 else ""))
        port = int(gcfg.get("first_port", 11435))
        with gpus.OllamaServers(self.gpu_list, self.cfg["ollama"], port, self.work / "cache") as servers:
            self.endpoints = servers.endpoints
            self.cfg = {**self.cfg, "ollama": servers.endpoints[0],
                        "transcribe": {**self.cfg["transcribe"], "device": self.gpu_list[0]["index"]}}
            self.log["gpus"] = [g["name"] for g in self.gpu_list]
            yield

    def _gpu_shares(self, name: str) -> list[list] | None:
        """[[GPU name, calls], ...] for a stage whose model calls were shared by several GPUs."""
        counts = getattr(self, "_split", {}).get(name) or []
        if len(counts) < 2 or not sum(counts):
            return None
        return [[g["name"].replace("NVIDIA GeForce ", ""), c] for g, c in zip(getattr(self, "gpu_list", []), counts)]

    def _print_split(self, name: str, what: str) -> None:
        shares = self._gpu_shares(name)
        if shares:
            console.print("  " + ", ".join(f"{gpu}: {c} {what}" for gpu, c in shares))

    @property
    def _parallel(self) -> bool:
        return len(self.gpu_list) > 1

    def _gpu(self, n: int) -> int | None:
        """nvidia-smi index of the n-th GPU of this run (None on a PC with one GPU: use the default)."""
        if not getattr(self, "_several_in_pc", False) or not self.gpu_list:
            return None
        return self.gpu_list[min(n, len(self.gpu_list) - 1)]["index"]

    # -- stages ---------------------------------------------------------------
    def run(self) -> Path:
        with self._gpus():
            return self._run()

    def _run(self) -> Path:
        cfg = self.cfg
        info = probe(self.source)
        streams = {name: info.find_audio(cfg["audio"].get(name)) for name in ("mic", "game", "party")}
        streams = {k: v for k, v in streams.items() if v is not None}
        if "mic" not in streams:
            raise SystemExit("No mic track found. Check the 'audio' section of your config.")
        console.print(f"[bold]{self.source.name}[/bold]: {info.duration / 3600:.2f} h, {info.width}x{info.height} "
                       f"@ {info.fps:.0f} fps, audio: {', '.join(f'{k}={v.title}' for k, v in streams.items())}")
        sr = cfg["audio"]["sample_rate"]

        def do_audio():
            paths = extract_audio_tracks(self.source, streams, self.work / "cache", sr)
            env = {name: signals.envelope_db(load_pcm(p), sr) for name, p in paths.items()}
            np.savez_compressed(self.work / "cache" / "envelopes.npz", **env)
            return {name: str(p) for name, p in paths.items()}
        pcm_paths = self._stage("audio", do_audio)
        env = dict(np.load(self.work / "cache" / "envelopes.npz"))

        # 2 GPUs: Whisper runs on the second card while the first one starts the kill feed scan.
        # When the transcript is done, the second card joins the kill feed scan.
        second_free = threading.Event()

        def do_transcript():
            tcfg = dict(cfg["transcribe"])
            if self._parallel:
                tcfg["device"] = self._gpu(1)
            regions = signals.voice_regions(env["mic"], tcfg["voice_margin_db"], tcfg["max_chunk_s"])
            speech_min = sum(e - s for s, e in regions) / 60
            costs.usage.transcribe_minutes += speech_min
            console.print(f"  {len(regions)} speech regions, {speech_min:.0f} min of speech")
            try:
                return transcribe_regions(load_pcm(Path(pcm_paths["mic"])), sr, regions, tcfg,
                                          progress=lambda d, n: console.print(f"  transcript {d}/{n}", end="\r"))
            finally:
                gc.collect()
                free_vram()
                second_free.set()

        kcfg, feed = cfg.get("killfeed") or {}, cfg["profile"].get("killfeed_region")
        scan_feed = bool(kcfg.get("enabled") and feed and cfg["vision"]["enabled"])
        if scan_feed and not kcfg.get("player_names"):
            console.print("[yellow]  kill feed scan skipped: set killfeed.player_names to your in-game name[/yellow]")
            scan_feed = False

        def do_killfeed():
            region = killfeed.feed_region(cfg["layout"]["gameplay"], feed)
            tiles = killfeed.extract_tiles(self.source, region, info.width, info.height, kcfg["every_s"],
                                           kcfg["panels_per_call"], self.work / "cache" / "killfeed")
            console.print(f"  {len(tiles)} kill feed images")
            ready = [None] + [second_free] * (len(self.endpoints) - 1)
            found = killfeed.read_tiles(tiles, cfg, progress=lambda d, n: console.print(f"  kill feed {d}/{n}", end="\r"),
                                        cache=self.work / "cache" / "killfeed" / "readings.jsonl",
                                        endpoints=self.endpoints, ready=ready, per_worker=self._split["killfeed"])
            self._print_split("killfeed", "kill feed images")
            return {"readings": found, "killers": killfeed.killer_stats(found)}

        segments, readings = [], []
        if self._parallel and scan_feed and cfg["transcribe"]["enabled"]:
            if self._json("transcript").exists() and "transcript" not in self.force and not self._dirty:
                second_free.set()  # transcript is cached: both cards scan the kill feed right away
            done = self._together([("transcript", do_transcript), ("killfeed", do_killfeed)])
            segments, readings = done["transcript"], done["killfeed"]["readings"]
        else:
            if cfg["transcribe"]["enabled"]:
                segments = self._stage("transcript", do_transcript)
            second_free.set()
            if scan_feed:
                readings = self._stage("killfeed", do_killfeed)["readings"]

        def do_candidates():
            sig = {name: signals.spike_score(db) for name, db in env.items()}
            sig["keywords"] = signals.keyword_score(segments, cfg["profile"].get("keywords"), len(env["mic"]))
            score = signals.combine(sig, cfg["signals"], cfg["signals"]["smooth_s"])
            np.save(self.work / "cache" / "score.npy", score)
            # the editor should be able to choose: about twice as many ~20 s moments as the target needs
            need = (0 if cfg.get("story_mode", {}).get("enabled", True) else
                    min(int(2 * cfg["editor"]["target_minutes"] * 60 / 20),
                        cfg["editor"].get("max_candidates", 60)))
            cands = signals.pick_candidates(score, segments, {"min_count": need, **cfg["candidates"]}, info.duration)
            if readings:
                kills = killfeed.player_kills(readings, kcfg["player_names"])
                groups = killfeed.group_kills(kills, kcfg["multikill_gap_s"])
                console.print(f"  kill feed: {len(kills)} kills in {len(groups)} moments")
                cands = killfeed.merge_into(cands, groups, segments, score, info.duration, cfg)
            return cands
        candidates = self._stage("candidates", do_candidates)
        console.print(f"  {len(candidates)} candidates")

        if cfg["vision"]["enabled"]:
            def do_vision():
                result = vision.analyse(info, [dict(c) for c in candidates], cfg,
                                        progress=lambda d, n: console.print(f"  {d}/{n}", end="\r"),
                                        endpoints=self.endpoints, per_worker=self._split["vision"])
                self._print_split("vision", "candidates")
                vision.rescore(result, cfg)
                return result
            candidates = self._stage("vision", do_vision)

        # Story-first workflow: render only selected narrative parts, no Shorts/captions.
        if cfg.get("story_mode", {}).get("enabled", True):
            return self._run_stories(info, streams, candidates, segments)

        fill = cfg["editor"].get("fill") is not False
        if self._json("edit").exists() and self._load("edit").get("fill", True) != fill:
            self.force.add("edit")  # the answer to "fill up?" changed: cut again

        def do_edit():
            clips, extra = editor.choose(candidates, cfg)
            return {"clips": clips, "fill": fill, **extra}
        edit = self._stage("edit", do_edit)
        total = sum(c["end"] - c["start"] for c in edit["clips"])
        console.print(f"  {len(edit['clips'])} clips, {total / 60:.1f} min")

        def do_render():
            out = self.work / "media"
            out.mkdir(exist_ok=True)
            jobs = list(enumerate(edit["clips"], 1))
            # one encoder per GPU of this run (with one GPU: the default encoder)
            encoders = [self._gpu(i) for i in range(max(1, len(self.gpu_list)))]
            return gpus.parallel_map(jobs, encoders, lambda job, gpu: render_clip(info, job[1], job[0], streams,
                                                                                   cfg, out, gpu=gpu))

        def do_shorts():
            out = self.work / "shorts"
            out.mkdir(exist_ok=True)
            gpu = self._gpu(1)  # 2 GPUs: Whisper and encoder on the second card
            # Select shorts from detected events, using the full timestamped transcript.
            try:
                picked = context_shorts.choose(candidates, segments, cfg, info.duration)
            except Exception as exc:
                console.print(f"[yellow]Context shorts failed ({exc}); using original selection[/yellow]")
                picked = shorts.pick(edit["clips"], cfg["shorts"]["count"], cfg["shorts"]["min_s"])
            if not picked:
                console.print("[yellow]Context AI selected 0 valid shorts; falling back to original clips.[/yellow]")
                picked = shorts.pick(edit["clips"], cfg["shorts"]["count"], cfg["shorts"]["min_s"])
            if not picked:
                console.print("[yellow]No clips meet shorts.min_s; no Shorts will be rendered.[/yellow]")
                return []
            console.print(f"  {len(picked)} shorts selected")
            pcm = load_pcm(Path(pcm_paths["mic"]))
            asr = load_asr({**cfg["transcribe"], **({"device": gpu} if gpu is not None else {})})
            words = [shorts.word_timestamps(pcm, sr, c["start"], c["start"] + min(c["end"] - c["start"],
                                            cfg["shorts"]["max_s"]), cfg, asr) for c in picked]
            del asr
            gc.collect()
            free_vram()
            result = []
            for n, (c, w) in enumerate(zip(picked, words), 1):
                ass = out / f"short_{n:02d}.ass"
                shorts.write_ass(w, ass, cfg["shorts"], shorts.caption_y(cfg))
                path = shorts.render(info, c, n, streams, ass, cfg, out, gpu=gpu)
                result.append({"file": str(path), "title": c.get("title", ""), "source_start": c["start"]})
            return result

        def do_metadata():
            meta = metadata.generate(edit["clips"], edit, cfg)
            (self.work / "youtube.txt").write_text(metadata.to_text(meta), encoding="utf-8")
            return meta

        late = [("render", do_render)]
        if cfg["shorts"]["enabled"]:
            late.append(("shorts", do_shorts))
        if cfg["metadata"]["enabled"]:
            late.append(("metadata", do_metadata))
        if self._parallel:
            # 2 GPUs: clips on both encoders, Shorts on the second card, metadata (LLM) on the first
            done = self._together(late)
        else:
            done = {name: self._stage(name, fn) for name, fn in late}
        rendered = done["render"]
        made_shorts = done.get("shorts", [])

        xml_path = self.work / f"{self.source.stem}_{cfg['export']['sequence_name_suffix']}.xml"
        self._stage("export", lambda: str(export_xmeml.write(
            rendered, list(streams), cfg, xml_path, f"{self.source.stem} {cfg['export']['sequence_name_suffix']}")))

        if cfg.get("thumbnails", {}).get("enabled") and edit["clips"]:
            self._stage("thumbnails", lambda: thumbnails.make(
                info, edit["clips"], [s["file"] for s in made_shorts], cfg, self.work / "thumbnails"))

        self._write_report(info, candidates, edit)
        return xml_path

    def _run_stories(self, info, streams, candidates, segments) -> Path:
        cfg = self.cfg
        def plan():
            return story_editor.choose(candidates, segments, cfg, info.duration,
                                       debug_output=str(self.work / "story_debug.json"), info=info)
        # Re-plan when upgrading the story algorithm, while keeping expensive
        # audio/transcript/candidate/vision caches intact.
        debug_path = self.work / "story_debug.json"
        if self._json("stories").exists():
            try:
                previous = json.loads(debug_path.read_text(encoding="utf-8"))
                if previous.get("version", 0) < 6:
                    self.force.add("stories")
            except (OSError, ValueError):
                self.force.add("stories")
        parts = self._stage("stories", plan)
        console.print(f"  {len(parts)} story parts selected")
        # Do not invent filler material if the model found no coherent story.
        if not parts:
            console.print("[yellow]No coherent stories found; no clips rendered.[/yellow]")
            self._write_report(info, candidates, {"clips": [], "video_title_ideas": []})
            return self.work

        def do_render():
            out = self.work / "media"
            out.mkdir(exist_ok=True)
            jobs = list(enumerate(parts, 1))
            encoders = [self._gpu(i) for i in range(max(1, len(self.gpu_list)))]
            return gpus.parallel_map(jobs, encoders,
                lambda job, gpu: render_clip(info, job[1], job[0], streams, cfg, out, gpu=gpu))

        rendered = self._stage("story_render", do_render)
        xml_path = self.work / f"{self.source.stem}_stories.xml"
        self._stage("story_export", lambda: str(export_xmeml.write(
            rendered, list(streams), cfg, xml_path, f"{self.source.stem} stories")))
        # A machine-readable edit decision is useful for reviewing story parts.
        self._write_report(info, candidates, {"clips": parts, "video_title_ideas": []})
        return xml_path

    def _write_report(self, info, candidates, edit) -> None:
        clips = edit["clips"]
        total = sum(c["end"] - c["start"] for c in clips)
        stages = self.log["stages"]
        lines = [
            f"# nightcut report: {self.source.name}",
            "",
            f"- Recording: {info.duration / 3600:.2f} h",
            f"- Candidates found: {len(candidates)}",
            *([f"- Kills read from the kill feed: {sum(c.get('kills', 0) for c in candidates)}"]
              if any("kills" in c for c in candidates) else []),
            f"- Clips in timeline: {len(clips)} ({total / 60:.1f} min)",
            f"- Total processing time: {costs.total_seconds(stages) / 60:.1f} min",
            f"- Highest VRAM use: {max((s['vram_peak_mib'] for s in stages.values()), default=0) / 1024:.1f} GB"
            + (" (per GPU)" if any(len(s.get("vram_by_gpu_mib") or []) > 1 for s in stages.values()) else ""),
            *([f"- GPUs: {' + '.join(self.log['gpus'])}"] if self.log.get("gpus") else []),
            *[f"- Shared {label}: " + ", ".join(f"{gpu} {c}" for gpu, c in stages[key]["by_gpu"])
              for key, label in (("killfeed", "kill feed images"), ("vision", "vision checks"))
              if stages.get(key, {}).get("by_gpu")],
            "",
            *[f"> ⚠️ {w}" for w in (edit.get("warning"),) if w],
            "",
            "## Title ideas",
            *[f"- {t}" for t in edit.get("video_title_ideas", [])],
            "",
            "## Clips",
            "| # | Source time | Length | Title | Why |",
            "|---|---|---|---|---|",
            *[f"| {n} | {int(c['start'] // 3600)}:{int(c['start'] % 3600 // 60):02d}:{int(c['start'] % 60):02d} "
              f"| {c['end'] - c['start']:.0f} s | {c.get('title', '')} | {c.get('reason', '')} |"
              for n, c in enumerate(clips, 1)],
            "",
            *(["## Thumbnail drafts", *[f"- `{p}`" for p in self._load("thumbnails")], ""]
              if self._json("thumbnails").exists() else []),
            "## Stages",
            "| Stage | Time | VRAM used (above baseline) |",
            "|---|---|---|",
            *[f"| {k} | {_time_text(v)} | {_vram_text(v)} |" for k, v in stages.items()],
            "",
            *costs.report_lines(costs.compute(stages, self.pricing), self.pricing, info.duration / 3600),
        ]
        (self.work / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

        meta = self._load("metadata") if self._json("metadata").exists() else {}
        made = self._load("shorts") if self._json("shorts").exists() else []
        s = summary.base(stages, self.pricing, self.source.name, info.duration / 3600,
                         edit.get("warning") or meta.get("warning"))
        s.update(mode="highlights", clips=len(clips), minutes=round(total / 60, 1),
                 target_minutes=self.cfg["editor"]["target_minutes"], shorts=len(made),
                 thumbnails=len(self._load("thumbnails")) if self._json("thumbnails").exists() else 0,
                 title=(meta.get("titles") or edit.get("video_title_ideas") or [""])[0],
                 xml=str(self.work / f"{self.source.stem}_{self.cfg['export']['sequence_name_suffix']}.xml"))
        summary.write(self.work, s)
