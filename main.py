import copy, csv, json, platform
from pathlib import Path
import numpy as np
import scipy
from scipy.optimize import linprog
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results'
TOL = 1e-7

def load_data():
    return json.loads((ROOT/'data.json').read_text(encoding='utf-8'))

def build_model(d):
    arcs = d['arcs']; nodes = d['sources']+d['hubs']+d['customers']
    A = np.zeros((len(nodes),len(arcs)))
    for j,a in enumerate(arcs):
        A[nodes.index(a['u']),j]=1
        A[nodes.index(a['v']),j]=-1
    k=len(d['sources'])
    return dict(c=np.array([a['cost'] for a in arcs]), A_ub=A[:k], b_ub=[d['supply'][v] for v in d['sources']], A_eq=A[k:], b_eq=[0]*len(d['hubs'])+[-d['demand'][v] for v in d['customers']], bounds=[(0,a['capacity']) for a in arcs])

def solve(d):
    return linprog(**build_model(d),method='highs-ds',options={'presolve':True,'disp':True,'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL})

def check(d,x):
    # Independent computation from arc endpoints, not model matrices or solver.
    x=np.asarray(x,dtype=float)
    if x.shape != (len(d['arcs']),):
        return dict(valid=False,cost=None,arcs=[],balances=[],error='Invalid flow vector shape')
    if not np.all(np.isfinite(x)):
        return dict(valid=False,cost=None,arcs=[],balances=[],error='Non-finite flow values')
    nodes=d['sources']+d['hubs']+d['customers']; net=np.zeros(len(nodes))
    rows=[]; cost=0.
    for a,f in zip(d['arcs'],x):
        net[nodes.index(a['u'])]+=f; net[nodes.index(a['v'])]-=f
        cost+=a['cost']*f
        rows.append(dict(arc=a['u']+'-'+a['v'],flow=float(f),capacity=a['capacity'],valid=bool(-TOL<=f<=a['capacity']+TOL)))
    balances=[]
    for v,n in zip(nodes,net):
        if v in d['supply']: target=d['supply'][v]; valid=-TOL<=n<=target+TOL
        else: target=-d['demand'].get(v,0); valid=abs(n-target)<=TOL
        balances.append(dict(node=v,net_out=float(n),target=target,valid=bool(valid)))
    return dict(valid=all(r['valid'] for r in rows+balances),cost=float(cost),arcs=rows,balances=balances)

def greedy(d):
    arcs=d['arcs']; cap=np.array([a['capacity'] for a in arcs],float); x=np.zeros(len(arcs)); supply=d['supply'].copy(); demand=d['demand'].copy()
    paths=sorted((a['cost']+b['cost'],i,j,a['u'],b['v']) for i,a in enumerate(arcs) for j,b in enumerate(arcs) if a['u'] in supply and a['v']==b['u'] and b['v'] in demand)
    for _,i,j,s,t in paths:
        f=min(cap[i],cap[j],supply[s],demand[t]); x[i]+=f; x[j]+=f; cap[i]-=f; cap[j]-=f; supply[s]-=f; demand[t]-=f
    return x

def write_csv(name,rows):
    if not rows:return
    with (OUT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def draw(d,x,name):
    G=nx.DiGraph(); G.add_edges_from((a['u'],a['v']) for a in d['arcs']); pos={}
    for i,layer in enumerate([d['sources'],d['hubs'],d['customers']]):
        for j,v in enumerate(layer):pos[v]=(i*4,(len(layer)-1)/2-j)
    fig,ax=plt.subplots(figsize=(12,7)); nx.draw_networkx_nodes(G,pos,node_size=1300,node_color='#dbeafe',edgecolors='#334155',ax=ax)
    labels={v: v+'\n'+str(d['supply'].get(v,d['demand'].get(v,''))) for v in G}; nx.draw_networkx_labels(G,pos,labels,font_size=10,ax=ax)
    for a,f in zip(d['arcs'],x):
        e=(a['u'],a['v']); color='#b91c1c' if abs(f-a['capacity'])<TOL else '#2563eb' if f>TOL else '#cbd5e1'
        nx.draw_networkx_edges(G,pos,edgelist=[e],edge_color=color,width=2 if f>TOL else 0.8,arrowsize=15,node_size=1300,ax=ax)
    nx.draw_networkx_edge_labels(G,pos,edge_labels={(a['u'],a['v']):f'{f:g}/{a["capacity"]}' for a,f in zip(d['arcs'],x)},font_size=8,label_pos=.3,rotate=False,ax=ax)
    ax.set_title(name+' — поток / ёмкость (ед./сутки)'); ax.axis('off'); fig.tight_layout(); fig.savefig(OUT/(name+'.png'),dpi=160); plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True); d=load_data(); base=solve(d); assert base.success
    usage=base.x/np.array([a['capacity'] for a in d['arcs']]); idx=int(np.argmax(usage)); arc=d['arcs'][idx]; print('Most loaded:',arc, 'utilization',usage[idx],flush=True)
    scenarios={'base':d}
    for name in ['capacity_minus20','failure','demand_plus20']:
        q=copy.deepcopy(d)
        if name=='capacity_minus20':q['arcs'][idx]['capacity']*=.8
        elif name=='failure':q['arcs'][idx]['capacity']=0
        else:q['demand']={k:round(v*1.2,8) for k,v in q['demand'].items()}
        scenarios[name]=q
    summary=[]; greedy_summary=[]; statuses={0:'optimal',1:'time/iteration limit',2:'infeasible',3:'unbounded',4:'numerical failure'}
    for name,q in scenarios.items():
        print('\nSCENARIO',name,flush=True); r=base if name=='base' else solve(q)
        row=dict(scenario=name,status=statuses.get(r.status,str(r.status)),cost='',delta='',verified='',iterations=r.nit,demand=sum(q['demand'].values()))
        if r.success:
            audit=check(q,r.x); assert audit['valid'] and abs(audit['cost']-r.fun)<=TOL
            audit['objective_error']=float(abs(audit['cost']-r.fun))
            row.update(cost=r.fun,delta=r.fun-base.fun,verified=True)
            write_csv(name+'_flows.csv',[dict(**a,flow=float(f),utilization=float(f/a['capacity']) if a['capacity'] else 0,shadow_price=float(p)) for a,f,p in zip(q['arcs'],r.x,r.upper.marginals)])
            (OUT/(name+'_check.json')).write_text(json.dumps(audit,indent=2)); draw(q,r.x,name)
        summary.append(row)
        # Compare the same greedy algorithm with HiGHS on every scenario.
        x=greedy(q); audit=check(q,x)
        prefix='greedy' if name=='base' else 'greedy_'+name
        (OUT/(prefix+'_check.json')).write_text(json.dumps(audit,indent=2),encoding='utf-8')
        write_csv(prefix+'_flows.csv',audit['arcs'])
        delivered=sum(-b['net_out'] for b in audit['balances'] if b['node'] in q['customers'])
        gap=(audit['cost']-r.fun)/r.fun*100 if audit['valid'] and r.success and abs(r.fun)>TOL else ''
        comparison=dict(scenario=name,lp_status=row['status'],lp_cost=row['cost'],greedy_valid=audit['valid'],greedy_cost=audit['cost'],demand=row['demand'],delivered=delivered,unmet_demand=max(0.,row['demand']-delivered),gap_pct=gap)
        greedy_summary.append(comparison); print('GREEDY',comparison,flush=True)
    write_csv('scenarios.csv',summary)
    write_csv('greedy_comparison.csv',greedy_summary)
    write_csv('node_prices.csv',[dict(node=v,shadow_price=float(p)) for v,p in zip(d['sources']+d['hubs']+d['customers'],np.r_[base.ineqlin.marginals,base.eqlin.marginals])]); print(summary)
    print('Python',platform.python_version(),'SciPy',scipy.__version__)

if __name__=='__main__':main()
