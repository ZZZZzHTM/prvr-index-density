"""Numerical implementation used by the reported experiment."""
import copy


def reverse(cell):
    out=copy.deepcopy(cell)
    if out['value'] is None:return out
    out['value']=-out['value']
    op=out['source']['operation']
    if op=='identity':out['source']['operation']='sign reversal of stored delta/loss only'
    elif op.startswith('sign reversal'):out['source']['operation']='identity'
    else:raise AssertionError(op)
    return out


def slot_sum(first,second):
    return dict(value=first['value']+second['value'],source=dict(operation='sum of index-cardinality components only',operands=[first,second]))


def quality_rows(result_file):
    import json
    from pathlib import Path
    data = json.loads(Path(result_file).read_text(encoding='utf8'))
    return [(row['model'],row['dataset'],row['Ward8']['value'],row['Ward16']['value'])
            for row in data['generality']]

if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser=argparse.ArgumentParser(description='Display the six-method Ward-8 and Ward-16 table.')
    parser.add_argument('result',nargs='?',type=Path,default=Path(__file__).resolve().parents[2]/'results'/'generality_six_methods'/'RESULTS.json')
    for row in quality_rows(parser.parse_args().result):
        print(*row,sep='\t')
