"""Auditable identifier reconciliation. No fuzzy or ambiguous automatic matches."""
from io import BytesIO
import json
import logging
from pathlib import Path
import pandas as pd
from .ingest_injuries import normalized_name
from .ingest_mlb import ensure_pitchers
from .utils import now, stable_id

LOG=logging.getLogger(__name__)
BASE='https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv'


def load_register(client):
    frames=[]
    for shard,body,error in client.many(list('0123456789abcdef'),lambda x:client.get(BASE.format(x),mutable=True)):
        if error:
            LOG.warning('Chadwick shard %s unavailable: %s',shard,error)
            continue
        df=pd.read_csv(BytesIO(body),dtype=str,low_memory=False)
        frames.append(df)
    if len(frames)!=16:
        raise RuntimeError('Incomplete Chadwick register; automatic name matches disabled')
    r=pd.concat(frames,ignore_index=True).fillna('')
    r['normalized_name']=(r.name_first+' '+r.name_last).map(normalized_name)
    return r


def prepare_candidates(client,cfg):
    path=cfg['report_path']/'poc_2024/unmatched_original_55.parquet'
    source=pd.read_parquet(path)
    register=load_register(client)
    output=[]
    for row in source.itertuples():
        matched=register.iloc[:0]
        method='unresolved'
        if pd.notna(row.fangraphs_id):
            matched=register[register.key_fangraphs.eq(str(row.fangraphs_id))]
            if len(matched)==1:method='chadwick_fangraphs_id'
        if matched.empty:
            matched=register[register.normalized_name.eq(normalized_name(row.player_name))]
            if len(matched)==1:method='unique_exact_register_name'
        usable=matched[matched.key_mlbam.ne('')]
        candidates=[]
        for person in usable.to_dict('records'):
            try:
                payload=client.json('https://statsapi.mlb.com/api/v1/people/'+str(int(float(person['key_mlbam']))))
                people=payload.get('people',[])
                if people:
                    candidates.append({'register':person,'mlb':people[0]})
            except Exception as exc:LOG.warning('Candidate metadata %s: %s',row.player_name,exc)
        # Search is evidence for review, never a fuzzy automatic match.
        if not candidates:
            try:
                found=client.json('https://statsapi.mlb.com/api/v1/people/search',{'names':row.player_name})
                for person in found.get('people',[]):candidates.append({'register':{},'mlb':person})
            except Exception as exc:LOG.warning('MLB name search %s: %s',row.player_name,exc)
        eligible=[]
        for c in candidates:
            p=c['mlb'];r=c['register']
            aliases=[p.get('fullName',''),p.get('firstLastName',''),p.get('useName','')+' '+p.get('lastName','')]
            exact=normalized_name(row.player_name) in {normalized_name(v) for v in aliases}
            dob=pd.to_datetime(p.get('birthDate'),errors='coerce')
            age=(row.surgery_date_lower-dob).days/365.2425 if pd.notna(dob) and pd.notna(row.surgery_date_lower) else None
            age_ok=age is not None and 12<=age<=60
            if pd.notna(row.age) and age is not None:age_ok=abs(age-float(row.age))<=2
            position_ok=not (row.position=='P' and p.get('primaryPosition',{}).get('abbreviation') not in ('P','TWP'))
            activity_start=pd.to_numeric(r.get('pro_played_first',''),errors='coerce')
            activity_end=pd.to_numeric(r.get('pro_played_last',''),errors='coerce')
            history_ok=bool(pd.notna(activity_start) and pd.notna(activity_end) and pd.notna(row.surgery_date_lower) and
                            activity_start-8<=row.surgery_date_lower.year<=activity_end+2)
            # Exact identifier overrides display-name changes; name-only cases require DOB and career compatibility.
            approve=(method=='chadwick_fangraphs_id' and age_ok) or (len(matched)==1 and exact and age_ok and position_ok and history_ok)
            if approve:eligible.append(c)
        accepted=eligible[0] if len(eligible)==1 else None
        output.append({'original_injury_id':row.injury_id,'player_name':row.player_name,
                       'source_tab':row.source_tab,'surgery_date_raw':row.surgery_date_raw,
                       'accepted_mlbam_id':accepted['mlb']['id'] if accepted else None,
                       'method':method if accepted else 'unresolved_review',
                       'candidates':candidates,'accepted_person':accepted['mlb'] if accepted else None,
                       'register_matches':len(matched)})
    path=cfg['report_path']/'injury_identity_candidates.json'
    path.write_text(json.dumps(output,indent=2,default=str))
    return output


def apply_reconciliation(db,client,cfg):
    path=cfg['report_path']/'injury_identity_candidates.json'
    candidates=json.loads(path.read_text()) if path.exists() else prepare_candidates(client,cfg)
    current=db.query('SELECT * FROM injuries')
    persons=[c['accepted_person'] for c in candidates if c['accepted_person']]
    ensure_pitchers(db,persons,'mlb_people')
    updates=[]
    for c in candidates:
        if c['accepted_mlbam_id'] is None:continue
        match=current[(current.player_name.map(normalized_name)==normalized_name(c['player_name'])) &
                      current.surgery_date_raw.eq(c['surgery_date_raw']) & current.source_tab.eq(c['source_tab'])]
        for r in match.itertuples():
            if pd.notna(r.pitcher_id) and r.pitcher_id!=c['accepted_mlbam_id']:continue
            flags=(r.quality_flags or '').replace('unmatched_mlbam','').strip(';')
            p=c['accepted_person']
            updates.append({'injury_id':r.injury_id,'pitcher_id':c['accepted_mlbam_id'],
                'identity_method':c['method'],'position':p.get('primaryPosition',{}).get('abbreviation'),
                'throws':r.throws or p.get('pitchHand',{}).get('code'),
                'quality_flags':flags or None})
    db.upsert('injuries',updates,['injury_id'])
    db.con.execute('''UPDATE injuries AS i SET repeat_procedure=EXISTS(SELECT 1 FROM injuries previous
        WHERE previous.pitcher_id=i.pitcher_id AND previous.confirmed_ucl_surgery
        AND previous.surgery_date_upper<i.surgery_date_lower)''')
    return candidates
