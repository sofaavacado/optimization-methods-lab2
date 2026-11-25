import json
from pathlib import Path
import numpy as np

def generate():
    rng = np.random.default_rng(10015)
    sources = ['S1','S2','S3']; hubs = ['H1','H2']; customers = [f'D{i}' for i in range(1,6)]
    supply = dict(zip(sources, [80,75,65]))
    demand = dict(zip(customers, map(int,rng.integers(25,36,5))))
    arcs = []
    for u,v in [(s,h) for s in sources for h in hubs]+[(h,d) for h in hubs for d in customers]:
        arcs.append(dict(u=u,v=v,capacity=int(rng.integers(35,61)),cost=int(rng.integers(2,13))))
    return dict(seed=10015,sources=sources,hubs=hubs,customers=customers,supply=supply,demand=demand,arcs=arcs)

if __name__ == '__main__':
    Path(__file__).with_name('data.json').write_text(json.dumps(generate(),indent=2),encoding='utf-8')
