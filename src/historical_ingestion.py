"""Season-resumable historical retrieval into the existing central database."""
from datetime import date, timedelta
import json
import logging
from . import ingest_mlb, ingest_statcast, ingest_injuries
from .utils import now, state

LOG = logging.getLogger(__name__)


def ingest(db, client, cfg, phase='all'):
    for year in range(cfg['start_date'].year, cfg['end_date'].year + 1):
        start = max(cfg['start_date'], date(year,1,1))
        end = min(cfg['end_date'], date(year,12,31))
        local = dict(cfg, start_date=start, end_date=end)
        LOG.info('HISTORICAL SEASON %s | %s to %s | phase=%s',year,start,end,phase)
        db.upsert('historical_checkpoints',[{'season':year,'phase':phase,'started_at':now(),'completed_at':None}],['season'])
        try:
            ingest_mlb.ingest_teams(db,client,year)
        except Exception as exc:
            state(db,'teams',year,'failed',detail=str(exc))
            LOG.warning('Season %s team source unavailable; preserving existing data: %s',year,exc)
        ingest_mlb.ingest_schedule(db,client,date(year,1,1),end)
        if phase in ('all','statcast'):
            ingest_statcast.ingest(db,client,local)
        if phase in ('all','support'):
            ingest_mlb.ingest_boxscores(db,client,date(year,1,1),end)
            ingest_mlb.ingest_transactions(db,client,date(year,1,1),end)
            ingest_mlb.ingest_previous_seasons(db,client,year,year)
            opening=db.scalar("SELECT min(game_date) FROM games WHERE season=? AND status='Final'",[year])
            if opening: ingest_mlb.ingest_rosters(db,client,opening)
        if phase in ('all','tracking'):
            ingest_mlb.ingest_pitch_tracking(db,client,local)
        count=db.scalar('SELECT count(*) FROM pitches WHERE game_date BETWEEN ? AND ?',[start,end])
        box=db.scalar('SELECT count(*) FROM boxscore_appearances b JOIN games g USING(game_id) WHERE season=?',[year])
        db.upsert('historical_checkpoints',[{'season':year,'phase':phase+'_complete','completed_at':now(),
                  'pitches':count,'appearances':box}],['season'])
        db.con.execute('CHECKPOINT')
        LOG.info('SEASON %s CHECKPOINT | %s raw pitches | %s official appearances',year,count,box)
        (cfg['report_path']/f'historical_sources_{year}_{phase}.json').write_text(json.dumps(client.records,indent=2,default=str))
        client.records.clear()
    if phase in ('all','support'):
        ingest_mlb.ingest_transactions(db,client,cfg['start_date']-timedelta(days=cfg['injury_link_lookback_days']),date(cfg['start_date'].year,1,1)-timedelta(days=1))
        ingest_mlb.ingest_people(db,client)
        ingest_injuries.ingest(db,client,cfg)
        ingest_mlb.ingest_people(db,client,missing_only=True)
    return {'passed':True,'ingestion_only':True}
