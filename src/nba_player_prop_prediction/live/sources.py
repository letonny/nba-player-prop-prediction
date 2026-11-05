"""NBA transport and normalized source contracts. Accessibility is not a license."""
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import requests
from pydantic import BaseModel, ConfigDict, Field, field_validator

from nba_player_prop_prediction.utils.io import write_json


class Game(BaseModel):
    model_config=ConfigDict(extra='forbid')
    game_id: str
    scheduled_tipoff: datetime
    home_id: str
    away_id: str
    home: str
    away: str
    status: Literal['scheduled','live','final','unknown']
    season: str='2026-27'
    @field_validator('scheduled_tipoff')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None: raise ValueError('Tipoff requires timezone')
        return v.astimezone(UTC)
    @field_validator('game_id','home_id','away_id')
    @classmethod
    def identifier(cls,v):
        if not v.isdigit(): raise ValueError('NBA identifier must be numeric')
        return v


class Player(BaseModel):
    player_id: str
    player_name: str
    team_id: str


class Availability(BaseModel):
    player_id: str
    game_id: str
    status: Literal['confirmed_active','probable','questionable','doubtful','confirmed_inactive','unknown']
    source: str = Field(min_length=1)
    published_at: datetime
    observed_at: datetime
    minutes_restricted: bool=False
    @field_validator('published_at','observed_at')
    @classmethod
    def aware(cls,v):
        if v.tzinfo is None: raise ValueError('Availability timestamp requires timezone')
        return v


class CachedNBA:
    def __init__(self,root:Path,timeout=15,retries=1,fetch=requests.get,sleep=time.sleep):
        self.root,self.timeout,self.retries,self.fetch,self.sleep=root,timeout,retries,fetch,sleep

    def get(self,url,ttl=300):
        if not url.startswith('https://cdn.nba.com/static/json/'):
            raise ValueError('Unsupported source host/path')
        key=hashlib.sha256(url.encode()).hexdigest()
        manifest=self.root/(key+'.json')
        old=json.loads(manifest.read_text()) if manifest.exists() else None
        now=datetime.now(UTC)
        if old and 0<=(now-datetime.fromisoformat(old['fetched_at'])).total_seconds()<ttl:
            return dict(old,provenance='cached')
        for attempt in range(self.retries+1):
            self.sleep(1.5+attempt*2)
            try:
                r=self.fetch(url,timeout=self.timeout)
                r.raise_for_status(); data=r.json()
                stamp=now.isoformat()
                snapshot=self.root/f'{key}-{time.time_ns()}.json'
                value={'source':url,'fetched_at':stamp,'provenance':'live_verified',
                       'snapshot':str(snapshot),'data':data}
                write_json(snapshot,value);write_json(manifest,value)
                return value
            except (requests.RequestException,ValueError):
                pass
        if old: return dict(old,provenance='cached',warning='Refresh failed; cached snapshot may be stale')
        return {'source':url,'provenance':'unavailable','fetched_at':None,'data':None,
                'warning':'Provider request failed'}


def schedule(payload):
    if not payload: return []
    rows=[]
    league=payload.get('leagueSchedule',{})
    for day in league.get('gameDates',[]):
        for g in day.get('games',[]):
            utc=g.get('gameDateTimeUTC')
            if not utc: continue
            rows.append(Game(game_id=str(g['gameId']),scheduled_tipoff=utc,
                home_id=str(g['homeTeam']['teamId']),away_id=str(g['awayTeam']['teamId']),
                home=g['homeTeam'].get('teamTricode') or g['homeTeam'].get('teamName','Unknown'),
                away=g['awayTeam'].get('teamTricode') or g['awayTeam'].get('teamName','Unknown'),
                status={1:'scheduled',2:'live',3:'final'}.get(g.get('gameStatus'),'unknown'),
                season=league.get('seasonYear','2026-27')).model_dump(mode='json'))
    if not rows: raise ValueError('Schedule contains no usable games')
    if len({g['game_id'] for g in rows})!=len(rows): raise ValueError('Duplicate schedule game')
    return rows


def roster(team_id,season='2026-27',timeout=15):
    """Current team roster endpoint; never infer rosters from old game logs."""
    from nba_api.stats.endpoints.commonteamroster import CommonTeamRoster
    if not str(team_id).isdigit(): raise ValueError('Invalid team ID')
    data=CommonTeamRoster(team_id=team_id,season=season,timeout=timeout).get_data_frames()[0]
    return [Player(player_id=str(r.PLAYER_ID),player_name=str(r.PLAYER),team_id=str(team_id)).model_dump()
            for r in data.itertuples()]


def final_results(payload):
    game=payload.get('game',{})
    if game.get('gameStatus')!=3: return None
    result={}
    for side in ['homeTeam','awayTeam']:
        for player in game.get(side,{}).get('players',[]):
            # played, not a zero box score, determines participation.
            played=player.get('played')
            participation='played' if played in [True,'1',1] else 'dnp' if played in [False,'0',0] else 'unknown'
            stats=player.get('statistics',{})
            values={k:stats.get(v) for k,v in {'points':'points','rebounds':'reboundsTotal',
                    'assists':'assists','threes_made':'threePointersMade'}.items()}
            if any(v is not None and (not isinstance(v,(int,float)) or v<0 or int(v)!=v) for v in values.values()):
                raise ValueError('Invalid final statistics')
            result[str(player['personId'])]={'participation':participation,'statistics':values}
    return {'game_id':str(game['gameId']),'players':result}


def boxscore_url(game_id):
    if not re.fullmatch(r'\d{10}',game_id): raise ValueError('Invalid game ID')
    return f'https://cdn.nba.com/static/json/liveData/boxscore/boxscore_{game_id}.json'


def fresh(timestamp,cutoff,max_hours):
    if not timestamp: return False
    t,c=pd.Timestamp(timestamp),pd.Timestamp(cutoff)
    if t.tzinfo is None or c.tzinfo is None: return False
    return pd.Timedelta(0)<=c-t<=pd.Timedelta(hours=max_hours)
