"""Дополнительные эксперименты ЛР2."""
import json
from time import perf_counter
from pathlib import Path
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from main import load_data, build_model, check, write_csv, OUT, TOL


def path_model(d):
    arcs = d['arcs']
    paths = [(i,j) for i,a in enumerate(arcs) for j,b in enumerate(arcs)
             if a['u'] in d['sources'] and a['v']==b['u'] and b['v'] in d['customers']]
    P = np.zeros((len(arcs),len(paths)))
    for k,(i,j) in enumerate(paths):
        P[i,k]=P[j,k]=1
    m=build_model(d)
    # Баланс хабов выполняется автоматически каждым полным путём.
    return paths,P,dict(c=m['c']@P,A_ub=np.vstack((m['A_ub']@P,P)),
        b_ub=np.r_[m['b_ub'],[a['capacity'] for a in arcs]],
        A_eq=m['A_eq'][len(d['hubs']):]@P,b_eq=m['b_eq'][len(d['hubs']):],bounds=(0,None))


def audit(d,x,objective,name,integer=False):
    a=check(d,x)
    a['objective_error']=abs(a['cost']-objective)
    if integer:
        a['integer_error']=float(np.max(np.abs(x-np.rint(x))))
    assert a['valid'] and a['objective_error']<=TOL and a.get('integer_error',0)<=TOL
    (OUT/(name+'_check.json')).write_text(json.dumps(a,indent=2),encoding='utf-8')


def run():
    OUT.mkdir(exist_ok=True); d=load_data(); m=build_model(d)
    opts={'presolve':True,'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL}
    lp=linprog(**m,method='highs-ds',options={**opts,'disp':True}); assert lp.success
    # Один рейс перевозит ровно одну тонну; перегрузка между рейсами в хабе разрешена.
    A=np.vstack((m['A_ub'],m['A_eq'])); lower=np.r_[[-np.inf]*len(m['b_ub']),m['b_eq']]; upper=np.r_[m['b_ub'],m['b_eq']]
    ip=milp(m['c'],integrality=np.ones(len(m['c'])),bounds=Bounds(0,[b[1] for b in m['bounds']]),constraints=LinearConstraint(A,lower,upper),options={'presolve':True,'disp':True,'mip_rel_gap':0.0}); assert ip.success
    audit(d,lp.x,lp.fun,'bonus_lp'); audit(d,ip.x,ip.fun,'bonus_milp',True)
    write_csv('bonus_integer_comparison.csv',[dict(model='LP',status='optimal',cost=lp.fun,integrality_gap_pct=0),dict(model='MILP',status='optimal',cost=ip.fun,integrality_gap_pct=(ip.fun-lp.fun)/lp.fun*100)])
    write_csv('bonus_integer_flows.csv',[dict(**a,lp_flow=float(x),milp_trips=int(round(y))) for a,x,y in zip(d['arcs'],lp.x,ip.x)])
    paths,P,p=path_model(d)
    # Прогрев, затем 30 чередующихся замеров только вызова решателя.
    for model in (m,p): assert linprog(**model,method='highs-ds',options=opts).success
    times=[[],[]]; results=[None,None]
    for k in range(30):
        for i in ([0,1] if k%2==0 else [1,0]):
            start=perf_counter(); r=linprog(**(m,p)[i],method='highs-ds',options=opts); times[i].append((perf_counter()-start)*1000); assert r.success; results[i]=r
    a,b=results; assert abs(a.fun-b.fun)<=TOL
    audit(d,a.x,a.fun,'bonus_node_arc'); audit(d,P@b.x,b.fun,'bonus_arc_path')
    write_csv('bonus_paths.csv',[dict(source=d['arcs'][i]['u'],hub=d['arcs'][i]['v'],customer=d['arcs'][j]['v'],flow=float(f),cost=d['arcs'][i]['cost']+d['arcs'][j]['cost']) for (i,j),f in zip(paths,b.x)])
    write_csv('bonus_path_flows.csv',[dict(**arc,node_arc=float(x),arc_path=float(y)) for arc,x,y in zip(d['arcs'],a.x,P@b.x)])
    rows=[]
    for name,model,r,t in zip(('node_arc','arc_path'),(m,p),results,times):
        rows.append(dict(model=name,variables=len(model['c']),inequalities=len(model['b_ub']),equalities=len(model['b_eq']),nonzeros=int(np.count_nonzero(model['A_ub'])+np.count_nonzero(model['A_eq'])),finite_upper_bounds=len(m['c']) if name=='node_arc' else 0,cost=r.fun,iterations=r.nit,median_ms=float(np.median(t)),min_ms=min(t),max_ms=max(t)))
    write_csv('bonus_formulations.csv',rows)
    write_csv('bonus_timings.csv',[dict(run=i+1,node_arc_ms=x,arc_path_ms=y) for i,(x,y) in enumerate(zip(*times))])
    print('BONUS MILP',ip.fun,'nodes',ip.mip_node_count,'gap',ip.mip_gap,'dual bound',ip.mip_dual_bound,flush=True); print('BONUS FORMULATIONS',rows,flush=True)

if __name__=='__main__': run()
