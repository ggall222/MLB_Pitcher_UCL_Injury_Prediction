"""Optional cache-only warming while another process owns the DuckDB writer lock."""
from datetime import date
import json
import logging
from .config import load_config
from .http import HttpClient
from .utils import date_chunks


def main():
    cfg=load_config();client=HttpClient(cfg)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    for year in range(cfg['start_date'].year,cfg['end_date'].year+1):
        ids=set();end=min(date(year,12,31),cfg['end_date'])
        for left,right in date_chunks(date(year,1,1),end):
            payload=client.json('https://statsapi.mlb.com/api/v1/schedule',{'sportId':1,'startDate':str(left),'endDate':str(right),'gameType':'R'},mutable=True)
            for day in payload.get('dates',[]):
                ids.update(g['gamePk'] for g in day.get('games',[]) if g['gameType']=='R' and g['status']['abstractGameState']=='Final')
        for i,(gid,body,error) in enumerate(client.many(sorted(ids),lambda gid:client.get(f'https://statsapi.mlb.com/api/v1/game/{gid}/boxscore'))):
            if error:logging.warning('Prefetch %s failed: %s',gid,error)
            if (i+1)%250==0:logging.info('Cached %s boxscores %s/%s',year,i+1,len(ids))
        (cfg['report_path']/f'prefetch_boxscores_{year}.json').write_text(json.dumps(client.records,indent=2))
        client.records.clear();logging.info('Boxscore cache season %s complete (%s games attempted)',year,len(ids))

if __name__=='__main__':main()
