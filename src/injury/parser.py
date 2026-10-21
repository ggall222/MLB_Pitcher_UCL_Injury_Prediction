"""Parse transaction lifecycle, anatomical evidence and dates without inventing onset."""
from datetime import date
import json
import re
from typing import Optional

from .taxonomy import STRUCTURES, REGION_PATTERNS, DIAGNOSES, UNKNOWN

IL = re.compile(r'\b(?:injured|disabled)\s+list\b|\b(?:IL|DL)\b', re.I)
LENGTH = re.compile(r'\b(7|10|15|60)[- ]day\b', re.I)
DATE_TEXT = r'(?:[A-Za-z]+\.?\s+\d{1,2}(?:,?\s+\d{4})?|\d{1,2}/\d{1,2}/\d{4}|\d{4}-\d{2}-\d{2})'
MONTHS = {name.lower(): i for i, name in enumerate(
    ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']) if i}
MONTHS.update({name[:3]: value for name, value in list(MONTHS.items())})


def as_date(value) -> Optional[date]:
    if value is None:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except (TypeError, ValueError):
        return None


def text_date(value: str, anchor: Optional[date]):
    value = value.strip().rstrip('.')
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}', value):
            return date.fromisoformat(value)
        if re.fullmatch(r'\d{1,2}/\d{1,2}/\d{4}', value):
            month, day, year = map(int, value.split('/'))
            return date(year, month, day)
        m = re.fullmatch(r'([A-Za-z]+)\.?\s+(\d{1,2})(?:,?\s+(\d{4}))?', value)
        if not m or (not m[3] and not anchor):
            return None
        year = int(m[3]) if m[3] else anchor.year
        result = date(year, MONTHS[m[1].lower()], int(m[2]))
        if not m[3] and result > anchor:
            result = date(year - 1, result.month, result.day)
        return result
    except (ValueError, KeyError):
        return None


def classify_transaction(description, type_code=None, type_description=None):
    text = str(description or '')
    related = bool(IL.search(text) or IL.search(str(type_description or '')))
    action = 'other'
    if related:
        if re.search(r'\b(?:activated|reinstated|reinstates|reactivated)\b', text, re.I):
            action = 'activation'
        elif re.search(r'\btransfer(?:red|s)?\b', text, re.I):
            action = 'transfer'
        elif re.search(r'\bplac(?:ed|es|ing)\b', text, re.I):
            action = 'placement'
        else:
            action = 'other_il_status'
    elif re.search(r'\brehab(?:ilitation)? assignment\b', text, re.I):
        action = 'rehab_assignment'
    lengths = [int(n) for n in LENGTH.findall(text)]
    return dict(il_related=related, il_action=action,
                il_length=lengths[-1] if lengths else None,
                il_from_length=lengths[0] if action == 'transfer' and len(lengths) > 1 else None)


def extract_injury_text(description):
    """Discard name/team/throwing-hand prefix; do not mistake RHP for right injury."""
    text = str(description or '')
    matches = list(IL.finditer(text))
    if not matches:
        return ''
    suffix = text[matches[-1].end():]
    suffix = re.sub(r'\bretroactive(?:\s+to)?\s+' + DATE_TEXT, '', suffix, flags=re.I)
    suffix = re.sub(r'^\s*(?:with\b|due to\b|because of\b)', '', suffix, flags=re.I)
    suffix = suffix.strip(' .;,')
    # A named concussion/COVID IL designation itself is explicit evidence.
    if not suffix:
        prefix = text[max(0, matches[-1].start() - 30):matches[-1].start()]
        if re.search(r'\bconcussion\b', prefix, re.I):
            suffix = 'Concussion'
        elif re.search(r'\bcovid(?:-19)?\b', prefix, re.I):
            suffix = 'COVID-19'
    return suffix


def laterality(text):
    if re.search(r'\bbilateral\b|\bboth (?:arms|shoulders|elbows|knees|legs|hands|ankles)\b', text, re.I):
        return 'Bilateral'
    found = set(m.group(1).lower() for m in re.finditer(r'\b(left|right)\b(?![- ]handed)', text, re.I))
    return next(iter(found)).title() if len(found) == 1 else 'Unknown'


def parse_injury(description):
    text = extract_injury_text(description)
    multi_player = len(re.findall(r'\b(?:RHP|LHP|TWP)\s+', str(description or ''))) > 1
    structures, regions, evidence = [], [], []
    # Negated or suspected diagnoses must not become confirmed UCL injuries.
    uncertain = bool(re.search(r'\b(?:possible|suspected|potential|rule out|evaluation|evaluated|precautionary)\b', text, re.I))
    excluded = bool(re.search(r'\b(?:no|without|ruled out|negative for)\s+(?:a\s+)?(?:ucl|tear|fracture|injury)', text, re.I))
    for name, region, pattern in STRUCTURES:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        if name == 'Labrum' and re.search(r'\bhip\b', text, re.I):
            region = 'Hip / Groin'
        if name == 'UCL' and re.search(r'\b(?:thumb|finger)\b', text, re.I) and not re.search(r'\belbow\b|tommy john', text, re.I):
            region = 'Hand / Wrist / Finger'
        if name == 'MCL' and re.search(r'\belbow\b', text, re.I):
            region = 'Elbow / UCL'
        if name.startswith('Flexor') and re.search(r'\belbow\b', text, re.I):
            region = 'Elbow / UCL'
        if name == 'Biceps' and re.search(r'\bshoulder\b', text, re.I):
            region = 'Shoulder'
        structures.append(name)
        if region not in regions:
            regions.append(region)
        evidence.append(match.group())
    broad = [(region, re.search(pattern, text, re.I)) for region, pattern in REGION_PATTERNS]
    for region, match in broad:
        if match and region not in regions:
            # A blister/laceration or local muscle fatigue is not an additional
            # anatomical region when its location is already explicit.
            specific_broad = any(m and r not in ('Other','Illness / non-musculoskeletal') for r,m in broad)
            if region == 'Other' and (regions or specific_broad) and re.fullmatch(r'blister|laceration', match.group(), re.I):
                continue
            if region == 'Illness / non-musculoskeletal' and (regions or specific_broad) and match.group().lower() == 'fatigue':
                continue
            regions.append(region)
            evidence.append(match.group())
    diagnoses = [name for name, pattern in DIAGNOSES if re.search(pattern, text, re.I)]
    if 'Stress Fracture' in diagnoses:
        diagnoses = [d for d in diagnoses if d != 'Fracture']
    surgery = bool(re.search(r'\b(?:surgery|surgical|postoperative|post-operative|reconstruction|repair|procedure)\b', text, re.I))
    surgery_type = None
    if surgery:
        if re.search(r'tommy john|ucl reconstruction|ulnar collateral ligament reconstruction', text, re.I):
            surgery_type = 'Tommy John / UCL reconstruction'
        elif re.search(r'internal[- ]brace', text, re.I):
            surgery_type = 'Internal brace repair'
        else:
            surgery_type = 'Other / unspecified surgery'
    region = regions[0] if regions else UNKNOWN
    if multi_player or excluded:
        region = UNKNOWN
    return dict(injury_text=text, injury_region=region,
                injury_structure='; '.join(structures) if structures and not excluded and not multi_player else None,
                injury_diagnosis=diagnoses[0] if diagnoses and not excluded and not multi_player else UNKNOWN,
                laterality=laterality(text) if not multi_player else 'Unknown',
                all_injury_regions=json.dumps(regions), all_injury_diagnoses=json.dumps(diagnoses),
                parser_evidence=json.dumps(evidence), multiple_injury_regions=len(regions) > 1,
                multiple_players_in_description=multi_player, diagnosis_uncertain=uncertain or excluded,
                surgery_flag=surgery, surgery_type=surgery_type,
                explicit_ucl=region == 'Elbow / UCL' and 'UCL' in structures and not uncertain and not excluded,
                parser_review_required=multi_player or excluded or uncertain or len(regions) > 1 or region == UNKNOWN)


def parse_dates(description, transaction_date, effective_date, maximum_retroactive_gap_days=31):
    tx, effective = as_date(transaction_date), as_date(effective_date)
    text = str(description or '')
    retro_match = re.search(r'\bretroactive(?:\s+to)?\s+(' + DATE_TEXT + ')', text, re.I)
    retro = text_date(retro_match[1], tx) if retro_match else None
    # Only explicit injury-language date statements count as documented onset.
    onset_match = re.search(r'\b(?:injured on|injury (?:occurred|sustained) on|sustained (?:the |an? )?injury on)\s+(' + DATE_TEXT + ')', text, re.I)
    onset = text_date(onset_match[1], tx) if onset_match else None
    candidates = [('explicit_documented_injury_date', onset), ('retroactive_description_date', retro),
                  ('mlb_effective_date', effective), ('transaction_date', tx)]
    # Old-year transcription errors occur in both description and API effective
    # date. Preserve them but do not turn them into year-long inferred absences.
    stale = lambda basis, d: bool(d and tx and basis in ('retroactive_description_date', 'mlb_effective_date')
                                 and (tx-d).days > maximum_retroactive_gap_days)
    valid = [(basis, d) for basis, d in candidates if d and (not tx or d <= tx) and not stale(basis, d)]
    basis, day = valid[0] if valid else ('unknown', None)
    invalid = bool((retro_match and retro is None) or (onset_match and onset is None) or
                   any(d and tx and d > tx for _, d in candidates) or any(stale(b, d) for b,d in candidates))
    surgery_match = re.search(r'\b(?:surgery|reconstruction|repair)\s+(?:on\s+)?(' + DATE_TEXT + ')', text, re.I)
    surgery_day = text_date(surgery_match[1], tx) if surgery_match else None
    if surgery_day and tx and surgery_day > tx:
        surgery_day = None
        invalid = True
    return dict(transaction_date=tx, effective_date=effective, retroactive_date=retro,
                explicitly_documented_onset_date=onset, injury_event_date=day,
                injury_event_date_basis=basis, date_parse_uncertain=invalid,
                surgery_date=surgery_day,
                event_date_known_at=tx, diagnosis_known_at=tx)
