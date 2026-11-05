"""Prepare dashboard historical data without fitting models or rewriting benchmarks."""
import json
from pathlib import Path

import pandas as pd

from nba_player_prop_prediction.utils.io import write_json


def historical_snapshot(root=Path('artifacts')):
    comparison=root/'probabilistic/comparison-v1'
    summary=json.loads((comparison/'summary.json').read_text())
    selected=pd.read_csv(root/'m4/backtest-v1/default85/selections.csv',dtype={'player_id':str,'game_id':str,'team_id':str,'opponent_id':str})
    # Last 250 actual retrospective selections for browsing; metrics use complete original run.
    selected=selected.sort_values(['game_date','probability'],ascending=[False,False]).head(250)
    stats=pd.read_parquet('data/processed/stats-v2.parquet').set_index('player_game_id')
    model_rows=[]
    for target,models in summary.items():
        name=json.loads((comparison/target/'validation_selection.json').read_text())['selected']
        report=models[name]
        model_rows.append({'target':target,'name':name,'version':report['version'],
                           'test':report['test'],'calibration':report['probabilities']['sigmoid'],
                           'periods':report['periods']})
        predictions=pd.read_parquet(comparison/target/name/'predictions.parquet').set_index('player_game_id')
        selected['projected_'+target]=selected.player_game_id.map(predictions['mean']*(1-predictions.gate))
    selected['predicted_minutes']=selected.player_game_id.map(stats.predicted_minutes)
    selected['provenance']='historical_backtest';selected['recorded']=False
    selected['grading_status']=selected.hit.map({True:'hit',False:'miss'})
    selected['prediction_id']=selected.candidate_id
    selected['generated_at']=None  # Backtest as-of dates are not forward creation timestamps.
    data=json.loads(selected.to_json(orient='records',date_format='iso'))
    write_json(root/'live/historical.json',{'data':data,'provenance':'historical_backtest',
        'as_of':str(selected.game_date.max()),'total':7720,'warning':'Retrospective 2025–26 selections; not live or forward-recorded forecasts'})
    write_json(root/'live/models.json',{'data':model_rows,'provenance':'historical_backtest'})
    return len(data)
