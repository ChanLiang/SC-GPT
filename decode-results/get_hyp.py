import json
import sys

suffix=sys.argv[1]
fi = f'dialogpt-ft-results-{suffix}.json'
fo = f'hyp-{suffix}'
results_from_gpt = json.load(open(fi))
with open(fo, 'w', encoding='utf-8') as w:
    for li in results_from_gpt:
        resp = li[0].split('<')[0]
        w.write(resp.strip() + '\n')
