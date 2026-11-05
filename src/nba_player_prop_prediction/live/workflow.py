"""Manual refresh, authorized imports, guarded inference and final-result grading."""
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.live.sources import Availability, CachedNBA, Game, Player, boxscore_url, final_results, fresh, schedule
from nba_player_prop_prediction.selection.engine import generate
from nba_player_prop_prediction.selection.policy import Policy, rank
from nba_player_prop_prediction.selection.tracking import Ledger
from nba_player_prop_prediction.utils.io import write_json, write_parquet

STATE=Path('artifacts/live/state.json')


def refresh(client=None):
    client=client or CachedNBA(Path('data/raw/live'))
    snapshot=client.get('https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json')
    try: games=schedule(snapshot['data']) if snapshot['data'] else []
    except (KeyError,ValueError,TypeError): games=[]
    result={'games':games,'players':[],'source':snapshot['source'],'fetched_at':snapshot['fetched_at'],
            'provenance':snapshot['provenance'] if games else 'unavailable','usage_authorized':False,
            'warning':snapshot.get('warning','NBA source usage permission is not established; roster refresh required')}
    write_json(STATE,result)
    return result


def import_source(path,output=STATE):
    """Import a timestamped authorized provider export; never promote illustrative inputs."""
    value=json.loads(Path(path).read_text())
    if value.get('provenance') not in ['live_verified','cached','illustrative_demo']:
        raise ValueError('Explicit source provenance required')
    if not value.get('source') or not value.get('fetched_at'): raise ValueError('Source and timestamp required')
    value['games']=[Game(**g).model_dump(mode='json') for g in value['games']]
    value['players']=[Player(**p).model_dump() for p in value.get('players',[])]
    ids=[p['player_id'] for p in value['players']]
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate or conflicting player roster IDs')
    value['import_sha256']=hashlib.sha256(Path(path).read_bytes()).hexdigest()
    write_json(output,value)
    return value


def contexts(state,history,date,cutoff,max_history_days=7):
    cutoff=pd.Timestamp(cutoff)
    if cutoff.tzinfo is None: raise ValueError('Cutoff must be timezone aware')
    if state.get('provenance') not in ['live_verified','cached'] or not state.get('usage_authorized'):
        raise ValueError('Verified authorized source required; illustrative data cannot produce forward forecasts')
    if not fresh(state.get('fetched_at'),cutoff,1): raise ValueError('Schedule is stale or post-cutoff')
    if not fresh(state.get('roster_fetched_at'),cutoff,24): raise ValueError('Roster is stale or post-cutoff')
    latest=history.loc[history.game_date<pd.Timestamp(cutoff.tz_convert('America/New_York').date()),'game_date'].max()
    if pd.isna(latest) or (cutoff.tz_localize(None).normalize()-latest).days>max_history_days:
        raise ValueError('Historical features are stale; refresh completed games first')
    rows=[]
    for raw in state['games']:
        g=Game(**raw)
        tip=pd.Timestamp(g.scheduled_tipoff)
        if tip.tz_convert('America/New_York').date().isoformat()!=date or g.status!='scheduled' or tip<=cutoff: continue
        for raw_player in state['players']:
            p=Player(**raw_player)
            if p.team_id not in [g.home_id,g.away_id]: continue
            home=p.team_id==g.home_id
            rows.append({'game_id':g.game_id,'game_date':date,'scheduled_tipoff':tip.isoformat(),
                         'season':g.season,'season_type':'Regular Season','team_id':p.team_id,
                         'opponent_id':g.away_id if home else g.home_id,'opponent':g.away if home else g.home,
                         'team':g.home if home else g.away,'home_away':'home' if home else 'away',
                         'player_id':p.player_id,'player_name':p.player_name})
    return rows


def availability_frame(path,cutoff):
    rows=[]
    if path:
        for value in json.loads(Path(path).read_text()):
            item=Availability(**value)
            if item.published_at>item.observed_at: raise ValueError('Invalid availability chronology')
            if not fresh(item.observed_at,cutoff,24) or pd.Timestamp(item.published_at)>=pd.Timestamp(cutoff): continue
            status={'confirmed_active':'available','confirmed_inactive':'inactive'}.get(item.status,item.status)
            rows.append(dict(player_id=item.player_id,game_id=item.game_id,status=status,source=item.source,
                             available_at=item.observed_at.isoformat(),trusted=True,minutes_restricted=item.minutes_restricted))
    return pd.DataFrame(rows) if rows else None


def forecast(date,cutoff,output,record=False,availability=None,policy=None):
    if output.exists(): raise ValueError('Use a new output path')
    state=json.loads(STATE.read_text())
    history=pd.read_parquet('data/processed/history.parquet')
    ctx=contexts(state,history,date,cutoff)
    forecasts=generate(history,ctx,cutoff,Path('artifacts/probabilistic/comparison-v1'),
                       Path('artifacts/probabilistic/oof-v2'),availability_frame(availability,cutoff))
    selected=rank(forecasts,policy or Policy(require_available=True))
    selected['source_snapshot']=state.get('import_sha256',state.get('source'))
    selected['source_fetched_at']=state['fetched_at']
    selected['roster_fetched_at']=state['roster_fetched_at']
    selected['provenance']='cached' if state['provenance']=='cached' else 'live_verified'
    selected['recorded']=False
    if record and len(selected):
        ledger=Ledger('artifacts/predictions.sqlite')
        for i,row in enumerate(json.loads(selected.to_json(orient='records',date_format='iso'))):
            selected.at[i,'prediction_id']=ledger.record(row)
        selected['recorded']=True;selected['provenance']='forward_recorded'
    write_parquet(output,selected)
    write_json(Path('artifacts/live/predictions.json'),{'data':json.loads(selected.to_json(orient='records',date_format='iso')),
        'as_of':datetime.now(UTC).isoformat(),'source_cutoff':cutoff,'provenance':'forward_recorded' if record else state['provenance']})
    return selected


def grade_pending(client=None,ledger_path=Path('artifacts/predictions.sqlite')):
    client=client or CachedNBA(Path('data/raw/live'))
    ledger=Ledger(ledger_path);rows=ledger.export();results=[];fetched={}
    for row in rows:
        if row['grading_status']!='pending': continue
        if row.get('market_available'):
            results.append({'id':row['prediction_id'],'status':'pending','reason':'Market settlement required'});continue
        game=row['game_id']
        if game not in fetched: fetched[game]=client.get(boxscore_url(game),ttl=300)
        source=fetched[game]
        if not source['data']: continue
        final=final_results(source['data'])
        if not final: continue
        if final['game_id']!=game: raise ValueError('Result game identity mismatch')
        player=final['players'].get(row['player_id'])
        if not player: continue  # Incomplete provider response is not a DNP.
        actual=player['statistics'].get(row['target'])
        if player['participation']=='played' and actual is None: continue
        status=ledger.grade(row['prediction_id'],actual,player['participation'],source['fetched_at'],source['snapshot'])
        results.append({'id':row['prediction_id'],'status':status})
    return results
