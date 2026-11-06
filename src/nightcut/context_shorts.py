"""Context-aware short selection using existing timestamped transcript segments."""
from __future__ import annotations

import json
from . import llm

SYSTEM = """Du schneidest eigenständig verständliche deutsche GTA-RP-Shorts.
Jeder Short ist eine abgeschlossene Geschichte, nicht nur eine Reaktion.
Ein Tod allein ist langweilig: zeige die erkennbare Ursache. Lachen allein ist
langweilig: zeige die erkennbare Pointe. Wenn der Kontext fehlt, verwerfe den Clip.
Wähle nur unterhaltsame Szenen und vermeide Routine, Leerlauf und unbegründete
Behauptungen. Der Start soll möglichst den Auslöser zeigen; das Ende die
Reaktion/Auflösung. Keine unnötigen Sekunden. Höchstens 180 Sekunden pro Clip.
Die Zeitangaben sind Sekunden in der ORIGINALAUFNAHME.
Wähle nur IDs aus der Eingabe. Antworte ausschließlich als JSON:
{"clips":[{"id":1,"start":100.0,"end":170.0,"title":"Kurzer Titel","reason":"Warum verständlich"}]}"""


def choose(candidates: list[dict], segments: list[dict], cfg: dict, duration: float) -> list[dict]:
    """Return context-expanded shorts, or [] if none can be justified."""
    scfg = cfg['shorts']
    limit = min(float(scfg.get('max_s', 180)), 180.0)
    minimum = float(scfg.get('min_s', 20))
    count = int(scfg.get('count', 3))
    if count <= 0 or not candidates:
        return []
    # A small, high-quality pool keeps local Ollama prompts manageable.
    pool = sorted(candidates, key=lambda c: c.get('final_score', c['score']), reverse=True)[:max(count * 3, 10)]
    listing = []
    windows = {}
    for c in pool:
        peak = float(c['peak'])
        lo = max(0.0, peak - 120.0)
        hi = min(duration, peak + 60.0)
        windows[c['id']] = (lo, hi)
        speech = [
            {'start': round(float(s['start']), 1), 'end': round(float(s['end']), 1),
             'text': str(s['text'])[:250]}
            for s in segments if s['end'] > lo and s['start'] < hi
        ]
        # Limit tokens per candidate while retaining both earlier and later dialogue.
        if len(speech) > 55:
            speech = speech[:30] + speech[-25:]
        listing.append({
            'id': c['id'], 'peak': peak, 'window': [round(lo, 1), round(hi, 1)],
            'event': c.get('vision', {}).get('event'),
            'visual_summary': c.get('vision', {}).get('summary', '')[:300],
            'speech': speech,
        })
    prompt = (f"Wähle höchstens {count} Shorts. Mindestlänge {minimum:.0f}s, "
              f"Höchstlänge {limit:.0f}s. Nur wenn der Kontext wirklich verständlich ist. "
              f"Daten: {json.dumps(listing, ensure_ascii=False)}")
    model = cfg['editor']['model']
    try:
        answer = llm.chat_json(cfg['ollama'], model, SYSTEM, prompt, temperature=0.2, max_tokens=2500)
    finally:
        llm.unload(cfg['ollama'], model)
    by_id = {c['id']: c for c in pool}
    selected = []
    used = set()
    for item in answer.get('clips', []):
        if not isinstance(item, dict):
            continue
        c = by_id.get(item.get('id'))
        if not c or c['id'] in used:
            continue
        lo, hi = windows[c['id']]
        try:
            start, end = float(item['start']), float(item['end'])
        except (ValueError, TypeError, KeyError):
            continue
        # Never trust model-provided timestamps without bounds checking.
        start, end = max(lo, start), min(hi, end)
        if end - start > limit:
            start = end - limit
        if end - start < minimum or start > c['peak'] or end < c['peak']:
            continue
        if any(start < old['end'] and end > old['start'] for old in selected):
            continue
        selected.append({**c, 'start': round(start, 2), 'end': round(end, 2),
                         'title': str(item.get('title', ''))[:80],
                         'reason': str(item.get('reason', ''))[:240]})
        used.add(c['id'])
        if len(selected) >= count:
            break
    return selected
