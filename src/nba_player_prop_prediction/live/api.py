"""Read-only local API. Page loads never fetch live data, train or run inference."""
import json
import os
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from nba_player_prop_prediction.selection.reports import forward_summary
from nba_player_prop_prediction.selection.tracking import Ledger


class Envelope(BaseModel):
    data: list[dict[str,Any]]
    provenance: str
    as_of: str | None=None
    warning: str | None=None
    total: int=0


class Health(BaseModel):
    status: str
    service: str='NBA local analytics'
    training_on_request: bool=False
    live_source: str


def create_app(root: Path | None=None):
    root=root or Path(os.getenv('NBA_ARTIFACTS_DIR','artifacts'))
    app=FastAPI(title='NBA Analytics',docs_url='/docs')
    def read(name,default):
        path=root/'live'/name
        if not path.exists(): return default
        try: return json.loads(path.read_text())
        except (ValueError,OSError): return default
    def ledger_rows():
        path=root/'predictions.sqlite'
        if not path.exists(): return []
        ledger=Ledger(path)
        try: return ledger.export()
        finally: ledger.db.close()
    def predictions(mode):
        if mode=='forward':
            return {'data':[dict(r,provenance='forward_recorded',recorded=True) for r in ledger_rows()],
                    'provenance':'forward_recorded'}
        result=read('historical.json' if mode=='historical' else 'predictions.json',
                    {'data':[],'provenance':'unavailable','warning':'No prediction snapshot available'})
        if mode=='current':
            today=pd.Timestamp.now(tz='America/New_York').date().isoformat()
            result=dict(result,data=[r for r in result['data'] if str(r.get('game_date',''))[:10]>=today])
            if not result['data']: result.update(provenance='unavailable',warning='No current forecasts; refresh data and generate pregame predictions')
        return result
    @app.get('/api/health',response_model=Health)
    def health():
        state=read('state.json',{})
        return Health(status='ok',live_source=state.get('provenance','unavailable'))
    @app.get('/api/games',response_model=Envelope)
    def games(date:str | None=Query(None,pattern=r'^\d{4}-\d{2}-\d{2}$')):
        state=read('state.json',{})
        rows=state.get('games',[])
        if date:
            rows=[g for g in rows if pd.Timestamp(g['scheduled_tipoff']).tz_convert('America/New_York').date().isoformat()==date]
        age=pd.Timestamp.now(tz='UTC')-pd.Timestamp(state['fetched_at']) if state.get('fetched_at') else None
        return Envelope(data=rows,provenance='cached' if age is not None else 'unavailable',
                        as_of=state.get('fetched_at'),total=len(rows),
                        warning='Schedule is stale; refresh required' if age is not None and age>pd.Timedelta(hours=1) else state.get('warning'))
    @app.get('/api/players',response_model=Envelope)
    def players(player_id:str | None=None):
        state=read('state.json',{})
        rows=state.get('players',[])
        return Envelope(data=[r for r in rows if not player_id or r['player_id']==player_id],
                        provenance='cached' if rows else 'unavailable',as_of=state.get('roster_fetched_at'),total=len(rows))
    @app.get('/api/players/{player_id}',response_model=Envelope)
    def player(player_id:str):
        if not player_id.isdigit(): raise HTTPException(400,'Invalid player identifier')
        path=root.parent/'data/processed/history.parquet'
        if not path.exists(): return Envelope(data=[],provenance='unavailable')
        h=pd.read_parquet(path,filters=[('player_id','==',player_id)])
        columns=['game_date','player_name','opponent','minutes','points','rebounds','assists','threes_made']
        h=h.sort_values('game_date').tail(20)
        return Envelope(data=json.loads(h[columns].to_json(orient='records',date_format='iso')),
                        provenance='historical_backtest',as_of=str(h.game_date.max()) if len(h) else None,total=len(h))
    @app.get('/api/predictions',response_model=Envelope)
    def forecasts(mode:Literal['current','historical','forward']='current',minimum:float=Query(0,ge=0,le=1),
                  target:str | None=None,game_id:str | None=None,player:str | None=None,
                  market_only:bool=False,limit:int=Query(250,ge=1,le=1000),offset:int=Query(0,ge=0)):
        result=predictions(mode)
        rows=[r for r in result['data'] if r.get('probability',0)>=minimum and
              (not target or r.get('target')==target) and (not game_id or r.get('game_id')==game_id) and
              (not player or player.lower() in str(r.get('player_name',r.get('player_id',''))).lower()) and
              (not market_only or r.get('market_available'))]
        rows.sort(key=lambda r:r.get('probability',0),reverse=True)
        return Envelope(data=rows[offset:offset+limit],provenance=result['provenance'],
                        as_of=result.get('as_of'),warning=result.get('warning'),total=len(rows))
    @app.get('/api/predictions/{prediction_id}',response_model=Envelope)
    def prediction(prediction_id:str,mode:Literal['current','historical','forward']='current'):
        result=predictions(mode)
        row=next((r for r in result['data'] if r.get('prediction_id',r.get('candidate_id'))==prediction_id),None)
        if row is None: raise HTTPException(404,'Prediction not found')
        return Envelope(data=[row],provenance=result['provenance'],total=1)
    @app.get('/api/models',response_model=Envelope)
    def models():
        result=read('models.json',{'data':[],'provenance':'unavailable'})
        return Envelope(**result,total=len(result['data']))
    @app.get('/api/performance')
    def performance():
        path=root/'m4/backtest-v1/default85/report.json'
        retrospective=json.loads(path.read_text()) if path.exists() else None
        return {'historical':retrospective,'historical_label':'Exploratory reused 2025–26 test data',
                'forward':forward_summary(ledger_rows()),'market_warning':'Statistical thresholds are not verified sportsbook offers'}
    return app


app=create_app()
