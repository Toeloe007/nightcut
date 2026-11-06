"""Select short, ordered source excerpts that together tell a GTA-RP story."""
from __future__ import annotations

import json
import re
from pathlib import Path
from . import llm

PICK_SYSTEM = """Du bist ein kritischer Video-Story-Editor für deutsches GTA-RP.
Dialoge sind die wichtigste Story-Evidenz: ein Streit, ein Witz, ein Funkspruch
oder eine überraschende Aussage können auch bei statischem Gameplay eine Story tragen.
Fehlende sichtbare Action widerlegt niemals einen eindeutig belegten Dialog.
Wähle NUR Geschichten mit erkennbarem Auslöser, spannender/komischer Entwicklung
und Auflösung. Eine Reaktion, ein Tod oder ein Krankenhausgespräch ALLEIN
sind keine Geschichte. Verbinde zusammengehörige Kandidaten (z.B. Verfolgung,
Zugunfall, Krankenhaus), auch wenn sie Minuten auseinander liegen. Erfinde
keine Ereignisse. Beurteile Bildbeschreibungen vorsichtig: sie können falsch sein.
Antworte ausschließlich JSON: {"stories":[{"ids":[9,10,11],"title":"...","why":"..."}]}.
Wenn mehrere zeitlich benachbarte Kandidaten durch konkrete Dialoghinweise zu einem
gemeinsamen Vorfall gehören, fasse sie zu einer Story zusammen, auch wenn
der Beginn des Vorfalls noch gesucht werden muss.
Gib {"stories":[]} zurück, wenn kein Zusammenhang belegbar ist.
Maximal 3 Geschichten, keine Überschneidung von Kandidaten."""

CUT_SYSTEM = """Du bist ein präziser Rohschnitt-Editor für GTA-RP-TikToks.
WICHTIG: Dialoge haben Vorrang vor Gameplay. Ein klarer verbaler Auslöser
(z.B. „Hinterher!“ oder „Wir halten den an“) muss im Einstieg hörbar sein,
auch wenn die Bilder nur ein fahrendes Auto zeigen. Bewahre Satzanfänge.
Erzeuge aus EINER Geschichte 2 bis 6 chronologische, NICHT überlappende
Ausschnitte. Einzeln als MP4 exportierbar; zusammengesetzt eine verständliche
Geschichte mit Beginn, Höhepunkt, Auflösung. Überspringe langweilige Fahrt,
Wartezeiten und Wiederholungen. Alle Zeiten sind ABSOLUTE Sekunden der Quelle.
Pflicht: Zeige zuerst, WIE die Verfolgung/Aktion begann, falls ein belegter
Story-Beginn mitgeliefert wird. Keine Eröffnung mitten im Höhepunkt.
Nach Unfall/Tod maximal EIN kurzer Epilog; keine wiederholten Krankenhaus-Szenen.
Wähle Schnitte anhand der ZEITGESTEMPELTEN Transkriptteile und bekannten
Ereignisankern. Erfinde keine visuell präzisen Schnittpunkte, die nicht aus
Daten hervorgehen. Setze Anfang und Ende möglichst an Satzgrenzen.
WICHTIG: Gesamtdauer aller Ausschnitte zusammen MAXIMAL 180 Sekunden,
jeder einzelne Ausschnitt 5 bis 75 Sekunden. Ein 10-Minuten-Ereignis muss
auf z.B. 20 s Beginn, 30 s Verfolgung, 40 s Unfall, 20 s Auflösung
gekürzt werden. Zwischen den Teilen sind große Zeitsprünge ERWÜNSCHT.
Gib NICHT die gesamte Geschichte in aufeinanderfolgenden langen Blöcken aus.
Berechne vor der Antwort die Summe der Schnittlängen und korrigiere sie.
Keine frei erfundenen Bildereignisse oder exakten Unfallzeitpunkte.
Mindestens ein Ausschnitt muss einen bekannten Ereignisanker enthalten.
Wenn der Ablauf nicht verständlich ist: {"parts":[]}.
Antworte ausschließlich JSON: {"parts":[{"start":123.0,"end":150.0,
"label":"Verfolgung beginnt"}]}"""


SETUP_SYSTEM = """Du untersuchst ein deutsches GTA-RP-Transkript VOR einem bekannten Ereignis.
Finde den frühesten BELEGTEN Beginn derselben Handlung (z.B. Funkspruch 'Verfolgung
aufgenommen', Sichtkontakt, Aufforderung anzuhalten, Fluchtbeginn). Eine allgemeine
Erwähnung einer Verfolgungsjagd oder ein anderes Ereignis ist KEIN Beleg.
Die Einträge enthalten absolute Sekunden. Wähle eine reale Zeit aus den Einträgen.
Antworte nur als JSON: {"start_s": 1234.5, "evidence": "...", "confidence": "high"}.
Wenn kein belegbarer Beginn vorhanden ist: {"start_s": null, "evidence": "nicht gefunden", "confidence": "low"}.
"""


def _setup_search(segments, first_anchor, cfg, entry, info=None):
    """Locate and (when video is available) visually verify the actual inciting incident."""
    earliest = max(0., first_anchor - float(cfg.get('story_mode', {}).get('lookback_s', 600)))
    before = [x for x in segments if x['end'] > earliest and x['start'] < first_anchor - 12]
    if not before:
        entry['setup_search'] = {'status': 'no_earlier_transcript'}
        return None
    # Multiple overlapping chronological blocks: do not discard early speech by stride.
    blocks = []
    for i in range(0, len(before), 55):
        group = before[i:i + 55]
        snippets = [{'s': round(x['start'], 1), 'e': round(x['end'], 1),
                     't': x['text'][:240]} for x in group]
        prompt = ('Story: ' + str(entry['proposal'].get('title', '')) +
                  '. Finde den TATSÄCHLICHEN BEGINN dieses Vorfalls, nicht spätere '
                  'Lageberichte oder Erwähnungen. Wenn er in diesem Block fehlt, '
                  'gib start_s:null zurück.\n' + json.dumps(snippets, ensure_ascii=False))
        try:
            response = _chat(cfg, SETUP_SYSTEM, prompt, 500)
        except Exception as exc:
            blocks.append({'range': [group[0]['start'], group[-1]['end']], 'error': repr(exc)})
            continue
        blocks.append({'range': [group[0]['start'], group[-1]['end']], 'response': response})
    entry['setup_search'] = {'status': 'searched', 'search_window': [earliest, first_anchor],
                             'blocks': blocks, 'visual_checks': []}
    proposals = []
    for block in blocks:
        response = block.get('response')
        if not isinstance(response, dict) or response.get('confidence') not in ('high', 'medium'):
            continue
        try:
            proposed = float(response['start_s'])
        except (ValueError, TypeError, KeyError):
            continue
        nearby = [x for x in before if abs(x['start'] - proposed) <= 8]
        if not nearby or not earliest <= proposed <= first_anchor - 35:
            continue
        seg = min(nearby, key=lambda x: abs(x['start'] - proposed))
        if all(abs(seg['start'] - q['start']) > 12 for q in proposals):
            proposals.append({'start': seg['start'], 'end': min(first_anchor - 10,
                              seg['start'] + 35), 'dialogue': ' '.join(x['text'] for x in before
                              if seg['start'] - 8 <= x['start'] <= seg['start'] + 40),
                              'story': entry['proposal'].get('title', ''),
                              'transcript_evidence': response.get('evidence', '')})
    proposals.sort(key=lambda x: x['start'])
    if not proposals:
        entry['setup_search']['status'] = 'no_transcript_start_found'
        return None
    # GTA-RP: an explicit spoken inciting incident is positive evidence.
    # A neutral gameplay frame is NOT negative evidence; visual checks are advisory.
    # The LLM has already tied each proposal to this specific story and supplied
    # a high/medium confidence, timestamped transcript quotation.
    selected = proposals[0]
    entry['setup_search']['dialogue_proposals'] = [
        {'start': round(p['start'], 2), 'evidence': p['transcript_evidence']}
        for p in proposals
    ]
    if info is not None and cfg.get('vision', {}).get('enabled', False):
        from . import vision
        checks = vision.verify_story_start(info, cfg, proposals)
        entry['setup_search']['visual_checks'] = checks
        for check in checks:
            result = check.get('vision') or {}
            if result.get('start_visible') is True and result.get('confidence') == 'high':
                selected = check
                entry['setup_search']['status'] = 'dialogue_and_visual_verified'
                break
    if entry['setup_search']['status'] == 'searched':
        entry['setup_search']['status'] = 'dialogue_verified_visual_advisory'
    start = float(selected['start'])
    entry['setup_search']['selected_s'] = round(start, 2)
    entry['setup_search']['selected_evidence'] = selected.get('transcript_evidence', '')
    return start


def _chat(cfg, system, prompt, tokens=2200):
    model = cfg['editor']['model']
    try:
        return llm.chat_json(cfg['ollama'], model, system, prompt,
                             temperature=0.15, max_tokens=tokens)
    finally:
        llm.unload(cfg['ollama'], model)


def choose(candidates: list[dict], segments: list[dict], cfg: dict,
           duration: float, debug_output: str | None = None, info=None) -> list[dict]:
    if not candidates:
        return []
    # Include every detected anchor; never rank a story solely by loudness.
    pool = sorted(candidates, key=lambda c: c['start'])[:60]
    overview = [{
        'id': c['id'], 'at_s': round(c['peak'], 1),
        'event': c.get('vision', {}).get('event', ''),
        'visual': c.get('vision', {}).get('summary', '')[:190],
        'dialogue_begin': c.get('transcript', '')[:650],
        'dialogue_end': c.get('transcript', '')[-900:],
    } for c in pool]
    debug = {'version': 6, 'candidate_count': len(pool), 'overview': overview,
             'pick_response': None, 'stories': [], 'status': 'started'}
    # The pipeline passes the exact work folder through an explicit argument below.
    def save_debug():
        if debug_file is not None:
            debug_file.parent.mkdir(parents=True, exist_ok=True)
            debug_file.write_text(json.dumps(debug, ensure_ascii=False, indent=2), encoding='utf-8')
    debug_file = Path(debug_output) if debug_output else None
    try:
        picked = _chat(cfg, PICK_SYSTEM,
                       'Kandidaten aus einer einzigen Aufnahme:\n' +
                       json.dumps(overview, ensure_ascii=False), 2800)
    except Exception as exc:
        debug['status'] = 'pick_error'
        debug['error'] = repr(exc)
        save_debug()
        raise
    debug['pick_response'] = picked
    save_debug()
    if not isinstance(picked, dict):
        debug['status'] = 'invalid_pick_response'
        save_debug()
        return []
    by_id = {c['id']: c for c in pool}
    result = []
    occupied: list[tuple[float, float]] = []
    for story_n, item in enumerate(picked.get('stories', [])[:3], 1):
        entry = {'story_index': story_n, 'proposal': item, 'status': 'started'}
        debug['stories'].append(entry)
        if not isinstance(item, dict):
            entry['status'] = 'invalid_story'
            save_debug()
            continue
        ids = item.get('ids', [])
        anchors = [by_id[i] for i in ids if isinstance(i, int) and i in by_id]
        if not anchors:
            entry['status'] = 'unknown_anchor_ids'
            save_debug()
            continue
        # Search for the *cause* of the story, not just the minutes before the climax.
        first_anchor = min(c['peak'] for c in anchors)
        setup_s = _setup_search(segments, first_anchor, cfg, entry, info)
        entry['setup_found'] = setup_s is not None
        entry['setup_source'] = entry.get('setup_search', {}).get('status')
        lo = max(0.0, (setup_s - 12) if setup_s is not None else
                 min(c['start'] for c in anchors) - 180)
        hi = min(duration, max(c['end'] for c in anchors) + 60)
        # Avoid enormous or unrelated spans in one planning call.
        if hi - lo > 1200:
            entry['status'] = 'story_window_too_long'
            entry['window'] = [lo, hi]
            save_debug()
            continue
        speech = [s for s in segments if s['end'] > lo and s['start'] < hi]
        # Preserve time ordering. If too many speech chunks, keep more around
        # known anchors, and sample the remainder across the whole span.
        if len(speech) > 150:
            chosen_idx = set()
            if setup_s is not None:
                # Keep EVERY utterance around the inciting incident, not a stride sample.
                chosen_idx.update(i for i, x in enumerate(speech)
                                  if setup_s - 10 <= x['start'] <= setup_s + 45)
            for c in anchors:
                near = sorted(range(len(speech)),
                              key=lambda i: abs((speech[i]['start'] + speech[i]['end']) / 2 - c['peak']))
                chosen_idx.update(near[:22])
            stride = max(1, len(speech) // 70)
            chosen_idx.update(range(0, len(speech), stride))
            speech = [speech[i] for i in sorted(chosen_idx)[:180]]
        transcript = [{'s': round(s['start'], 1), 'e': round(s['end'], 1),
                       't': s['text'][:180]} for s in speech]
        evidence = [{'at': round(c['peak'], 1),
                     'event': c.get('vision', {}).get('event'),
                     'visual': c.get('vision', {}).get('summary', '')[:200]}
                    for c in anchors]
        setup_instruction = (
            f'DIALOGBASIERTER STORY-BEGINN bei {setup_s:.1f}s: Der ERSTE Clip MUSS '
            f'diesen Zeitpunkt und die auslösende Aussage hörbar enthalten. '
            f'Beginne spätestens {setup_s:.1f}s (nicht erst 12s danach). '
            'Mindestens ein weiterer Clip zeigt die Entwicklung VOR dem Höhepunkt. '
            if setup_s is not None else
            'Kein belegter Story-Beginn im Transkript gefunden. '
            'Erfinde keinen Anfang; falls die Handlung unverständlich bleibt, gib parts:[] zurück. '
        )
        prompt = (f"Story: {item.get('title','')}. Zeitfenster [{lo:.1f},{hi:.1f}]. "
                  'Maximal 180 Sekunden Gesamtmaterial. '
                  'Bevorzugte Dramaturgie: Einstieg 15-30s, Entwicklung 15-35s, '
                  'Höhepunkt 20-45s, NACHWIRKUNGEN INSGESAMT MAXIMAL 25s. '
                  'Keine mehreren Krankenhaus-/Tot-Szenen. '
                  + setup_instruction +
                  json.dumps({'anchors': evidence, 'speech': transcript}, ensure_ascii=False))
        entry['attempts'] = []
        parts = []
        valid = False
        for attempt in range(3):
            try:
                plan = _chat(cfg, CUT_SYSTEM, prompt, 2800)
            except Exception as exc:
                entry['status'] = 'cut_error'
                entry['error'] = repr(exc)
                save_debug()
                raise
            trial = {'attempt': attempt + 1, 'cut_response': plan, 'rejected_parts': []}
            entry['attempts'].append(trial)
            entry['cut_response'] = plan
            save_debug()
            if not isinstance(plan, dict) or not isinstance(plan.get('parts'), list):
                trial['reason'] = 'invalid_cut_response'
                prompt += '\nDeine vorige Antwort war kein gültiges JSON-Objekt mit parts-Liste. Korrigiere sie.'
                continue
            parts = []
            for raw in plan['parts'][:6]:
                try:
                    a, b = float(raw['start']), float(raw['end'])
                except (TypeError, ValueError, KeyError):
                    trial['rejected_parts'].append({'part': raw, 'reason': 'invalid_times'})
                    continue
                if not (lo <= a < b <= hi) or not (5 <= b - a <= 75):
                    trial['rejected_parts'].append({'part': raw, 'reason': 'out_of_bounds_or_duration'})
                    continue
                if parts and a < parts[-1]['end']:
                    trial['rejected_parts'].append({'part': raw, 'reason': 'overlap_or_wrong_order'})
                    continue
                if any(a < e and b > s for s, e in occupied):
                    trial['rejected_parts'].append({'part': raw, 'reason': 'overlap_with_other_story'})
                    continue
                parts.append({'start': round(a, 2), 'end': round(b, 2),
                              'label': str(raw.get('label', 'Szene'))[:65]})
            total = round(sum(p['end'] - p['start'] for p in parts), 2)
            anchor_covered = any(p['start'] <= c['peak'] <= p['end']
                                 for p in parts for c in anchors)
            setup_covered = (setup_s is None or
                             (bool(parts) and parts[0]['start'] <= setup_s + 1
                              and parts[0]['end'] >= setup_s))
            # Limit material after the known climax; the ending is an epilogue,
            # not a second story. Use the earliest anchor as the beginning of
            # the climax region and the latest anchor as the aftermath region.
            last_anchor = max(c['peak'] for c in anchors)
            aftermath_s = sum(max(0.0, p['end'] - max(p['start'], last_anchor - 12))
                              for p in parts)
            aftermath_ok = aftermath_s <= 30
            trial['setup_covered'] = setup_covered
            trial['aftermath_s'] = round(aftermath_s, 1)
            valid = (2 <= len(parts) <= 6 and total <= 180
                     and not trial['rejected_parts'] and anchor_covered
                     and setup_covered and aftermath_ok)
            trial['accepted_parts'] = parts
            trial['total_s'] = total
            trial['reason'] = ('accepted' if valid else
                               'invalid_part_count_or_total_length_or_anchor')
            entry['accepted_parts'] = parts
            entry['rejected_parts'] = trial['rejected_parts']
            save_debug()
            if valid:
                break
            prompt += ('\nDEIN VORHERIGER SCHNITT WURDE ABGELEHNT. '
                       f'Fehler: {json.dumps(trial["rejected_parts"], ensure_ascii=False)}. '
                       f'Gültige Teile: {len(parts)}, Gesamt: {total} s, '
                       f'Ereignisanker enthalten: {anchor_covered}. '
                        f'Story-Einstieg enthalten: {setup_covered}. '
                        f'Nachwirkungen: {aftermath_s:.1f}s (max. 30s). '
                       'Gib einen NEUEN, DEUTLICH KÜRZEREN Schnittplan zurück. '
                       'Ziel 90-150 s Gesamtmaterial, 3-5 Teile, je 10-45 s. '
                        'WICHTIG: erster Teil muss die gefundene Einstiegsszene enthalten. '
                        'Nachwirkungen höchstens 25s, nicht mehrere Krankenhausclips. '
                       'Zwischen den Teilen Zeit überspringen! '
                       'Keine aneinandergrenzenden 90-Sekunden-Blöcke. '
                       'Benutze die Zeitstempel im Transkript und die Anker.')
        if not valid:
            entry['status'] = 'cut_validation_failed_after_3_attempts'
            save_debug()
            continue
        # Editorial safety handles: retain up to 20 seconds BEFORE the detected
        # dialogue-based setup. This runs AFTER cut validation, so the LLM's
        # established story selection and all later parts stay unchanged.
        # Respect the existing 75-second-per-part and 180-second-total limits.
        if setup_s is not None and parts:
            original_start = parts[0]['start']
            target_start = max(0.0, setup_s - 20.0)
            max_by_part = parts[0]['end'] - 75.0
            max_by_total = original_start - max(0.0, 180.0 - total)
            new_start = round(max(target_start, max_by_part, max_by_total, 0.0), 2)
            # Do not expand into a previously accepted story's time range.
            for occupied_start, occupied_end in occupied:
                if occupied_end > new_start and occupied_start < original_start:
                    new_start = max(new_start, occupied_end)
            new_start = round(min(original_start, new_start), 2)
            if new_start < original_start:
                parts[0]['start'] = new_start
            entry['setup_preroll'] = {
                'requested_s': 20.0,
                'detected_setup_s': round(setup_s, 2),
                'original_first_start_s': original_start,
                'final_first_start_s': parts[0]['start'],
                'added_s': round(original_start - parts[0]['start'], 2),
            }
            entry['accepted_parts'] = parts
            trial['accepted_parts'] = parts
            trial['total_s'] = round(sum(p['end'] - p['start'] for p in parts), 2)
        entry['status'] = 'accepted'
        save_debug()
        title = str(item.get('title', f'Story {story_n}'))[:90]
        for part_n, part in enumerate(parts, 1):
            # render_clip consumes start/end/title; other candidate metadata is optional.
            result.append({**part, 'id': len(result) + 1,
                           'peak': (part['start'] + part['end']) / 2,
                           'score': max(c.get('score', 0) for c in anchors),
                           'title': f'S{story_n:02d} P{part_n:02d} - {part["label"]}',
                           'reason': title, 'story': story_n,
                           'story_title': title, 'part': part_n})
            occupied.append((part['start'], part['end']))
    debug['status'] = 'complete'
    debug['selected_part_count'] = len(result)
    save_debug()
    return result
