"""Local-only data and prediction operations; no administrative HTTP endpoints."""
import argparse
import json
from pathlib import Path

from nba_player_prop_prediction.live.snapshot import historical_snapshot
from nba_player_prop_prediction.live.workflow import forecast, grade_pending, import_source, refresh
from nba_player_prop_prediction.selection.policy import Policy


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    sub.add_parser('snapshot');sub.add_parser('refresh');sub.add_parser('grade')
    imp=sub.add_parser('import-source');imp.add_argument('path',type=Path)
    gen=sub.add_parser('predict');gen.add_argument('--date',required=True);gen.add_argument('--cutoff',required=True)
    gen.add_argument('--output',type=Path,required=True);gen.add_argument('--record',action='store_true')
    gen.add_argument('--availability',type=Path);gen.add_argument('--minimum',type=float,default=.85)
    gen.add_argument('--allow-unknown-availability',action='store_true')
    history=sub.add_parser('refresh-history');history.add_argument('--season',default='2026-27')
    history.add_argument('--start',required=True);history.add_argument('--end',required=True)
    history.add_argument('--authorized-source',action='store_true',required=True)
    roster=sub.add_parser('refresh-roster');roster.add_argument('--team-id',required=True)
    roster.add_argument('--authorized-source',action='store_true',required=True)
    args=parser.parse_args()
    if args.command=='snapshot': print(historical_snapshot())
    elif args.command=='refresh':
        result=refresh();print(json.dumps({k:v for k,v in result.items() if k not in ['games','players']},indent=2))
        if result['provenance']=='unavailable': raise SystemExit(2)
    elif args.command=='import-source': print(len(import_source(args.path)['games']))
    elif args.command=='predict':
        result=forecast(args.date,args.cutoff,args.output,args.record,args.availability,
                        Policy(minimum=args.minimum,require_available=not args.allow_unknown_availability))
        print(f'{len(result)} forecasts; {int(result.selected.sum())} selected')
    elif args.command=='refresh-history':
        from nba_player_prop_prediction.config import Settings
        from nba_player_prop_prediction.ingestion.collector import Collector
        from nba_player_prop_prediction.ingestion.expansion import assemble
        Collector(Settings.from_env()).collect(args.season,'Regular Season',args.start,args.end,refresh=True)
        assemble(Settings.from_env(),Path('data/processed/history.parquet'))
    elif args.command=='refresh-roster':
        from datetime import UTC,datetime
        from nba_player_prop_prediction.live.sources import roster
        from nba_player_prop_prediction.utils.io import write_json
        path=Path('data/raw/live')/f'roster-{args.team_id}.json'
        value={'players':roster(args.team_id),'source':'nba_api.CommonTeamRoster',
               'fetched_at':datetime.now(UTC).isoformat()}
        write_json(path,value);print(path)
    else: print(json.dumps(grade_pending(),indent=2))


if __name__=='__main__': main()
