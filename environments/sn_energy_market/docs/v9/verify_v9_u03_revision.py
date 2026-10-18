"""Current v9 documentation reference. No production imports or edits.

Reuses the unchanged market/physics/cash reference, then independently evaluates
the new capacity assessment. Numerical policy comparisons use stated candidates.
"""
from pathlib import Path
from collections import Counter
from dataclasses import replace, asdict
import hashlib
import importlib.util
import json
import math
import random
import re
import sys
from urllib.parse import unquote

BASE=Path(__file__).resolve().parent
ROOT=BASE.parents[3]
SEED=10142026

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def canonical(tex): return re.sub(r'\s+','',re.sub(r'\\tag\{\d+\}','',tex))
def digest(text): return hashlib.sha256(text.encode()).hexdigest()
def close(a,b=0.0): assert math.isclose(a,b,rel_tol=1e-9,abs_tol=1e-9),(a,b)

def load_reference():
    path=BASE/'verify_v9_mechanisms.py'
    assert sha(path)=='d732f140462839469c677fcab425f96aa70c9ab6cf2304f2fc6d425ef0ebed97'
    spec=importlib.util.spec_from_file_location('v9_u03_historical_physics',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

def assessment(baseline,due,capacity,dr_cap,net_dr,event,price,dt,rate=2.0):
    credit=min(dr_cap,net_dr) if event and price>0 else 0.0
    gap=max(0.0,baseline+due-capacity-credit)
    return credit,gap,rate*gap*dt

def revised_result(orders,s,result,rate=2.0,late_energy=None):
    baseline=sum(o.baseline for o in orders)
    due=sum(o.due_attempt for o in orders)
    capacity=result['der_capacity']+result['uc_capacity']
    credit,gap,cost=assessment(baseline,due,capacity,s.dr_cap,result['net_dr'],s.event,s.dr_price,s.dt,rate)
    late_energy=sum(c['due_attempt_unserved'] for c in result['consumers'].values())*s.dt if late_energy is None else late_energy
    service=cost+s.due_penalty*late_energy
    reward=result['cash']-result['degradation_cost']-result['curtail_cost']+result['benefit']-service
    return {'baseline_kw':baseline,'due_kw':due,'announced_capacity_kw':capacity,'credited_DR_kw':credit,
            'capacity_gap_kw':gap,'capacity_assessment':cost,'late_energy_kwh':late_energy,
            'service_cost':service,'reward':reward,'cash':result['cash']}

def check_document():
    text=(BASE/'modeling_complete_zh.md').read_text(encoding='utf-8')
    baseline=json.loads((BASE/'u03_revision_baseline.json').read_text(encoding='utf-8'))
    assert re.findall(r'^## (\d+)\. ',text,re.M)==[str(i) for i in range(1,27)]
    assert len(re.findall(r'^# 第.+部分',text,re.M))==8
    displays=re.findall(r'\$\$([\s\S]*?)\$\$',text)
    numbered={int(m.group(1)):canonical(tex) for tex in displays if (m:=re.search(r'\\tag\{(\d+)\}',tex))}
    assert list(numbered)==list(range(1,31))
    for number,expected in baseline['unchanged_numbered_relations'].items():
        assert digest(numbered[int(number)])==expected['relation_sha256'],number
    plain=Counter(digest(canonical(tex)) for tex in displays if not re.search(r'\\tag\{\d+\}',tex))
    assert not (Counter(baseline['unchanged_unnumbered_relations'])-plain)
    codes=Counter(digest(c.strip()) for c in re.findall(r'```[^\n]*\n([\s\S]*?)```',text))
    assert not (Counter(baseline['unchanged_code_examples'])-codes)
    for name,expected in baseline['historical_documentation_QA_sha256'].items(): assert sha(BASE/name)==expected,name
    order=['## 3. UC 的 Observation','## 4. UC 的 Action','## 5. UC 的 Reward',
           '## 6. Consumer 的 Observation','## 7. Consumer 的 Action','## 8. Consumer 的 Reward',
           '## 9. 市场','## 10. 环境']
    assert [text.index(x) for x in order]==sorted(text.index(x) for x in order)
    compact=re.sub(r'\s+','',text)
    required=[r'\sum_i(p_i^{base}+p_i^{due})-p^{DER,cap}-p^{UC,cap}',
              r'\mathbf{1}_{\{\chi=1,\T^{DR}>0\}}\min(p^{DR,cap},p^{DR,net})',
              r'V_{\omega_U}(o_t^U)',r'x_t^U=o_t^U',r'\sum_iq_i^{(0)}',r'H+20']
    for x in required: assert x in compact,x
    assert '按以下顺序生成联合可行行动' not in text
    assert 'C^{UC,service}=\\lambda_n\\sum_i p_i^{lost}' not in text
    heading_ids=re.findall(r'^### (\d+\.\d+) ',text,re.M)
    assert len(heading_ids)==len(set(heading_ids))
    for m in re.finditer(r'第\s*(\d+)—(\d+)\s*节',text): assert int(m[1])<=int(m[2]),m.group()
    links=[]
    for path in BASE.glob('*.md'):
        for link in re.findall(r'\[[^\]]*\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if re.match(r'^(?:https?:|#)',link): continue
            target=(path.parent/unquote(link.split('#')[0])).resolve()
            # This report is the output of this run and is checked after writing.
            assert target.exists() or target==BASE/'u03_revision_validation.json',(path.name,link)
            links.append((path.name,link))
    return {'parts':8,'chapters':26,'numbered_equations':30,'display_formulas':len(displays),
            'unchanged_numbered_relations':len(baseline['unchanged_numbered_relations']),
            'unchanged_unnumbered_relations':len(baseline['unchanged_unnumbered_relations']),
            'code_examples_preserved':len(baseline['unchanged_code_examples']),
            'old_QA_programs_unchanged':len(baseline['historical_documentation_QA_sha256']),
            'subject_order':order,'local_links_checked':len(links),'number_map':baseline['number_map']}

def binary_response(ref,supply,rate):
    options=[]
    for cut in [0.0,4.0]:
        orders=[ref.Order('A',4,curtail=cut,uc_tariff=0.4)]
        original=ref.run(orders,supply)
        c=original['consumers']['A']
        consumer_cost=c['net_transfer']+0.75*cut*cut/4*0.25+2*c['native_lost']*0.25
        revised=revised_result(orders,supply,original,rate)
        options.append({'supply':supply.grid_offer,'DR_cap':supply.dr_cap,'cut':cut,
                        'consumer_cost':consumer_cost,**revised})
    return min(options,key=lambda x:(x['consumer_cost'],x['cut']))

def continuous_response(supply,dr_cap,kappa=0.5):
    knots=sorted(set([0.0,4.0,4.0-supply,dr_cap]))
    candidates=list(knots)
    for left,right in zip(knots[:-1],knots[1:]):
        mid=(left+right)/2
        marginal=(1.0 if mid<4.0-supply else 0.4)+(0.5 if mid<dr_cap else 0.0)
        stationary=marginal*4/(2*kappa)
        if left<=stationary<=right:candidates.append(stationary)
    def cost(cut):
        served=min(supply,4-cut);lost=max(0.0,4-cut-served)
        return .25*(.4*served+lost+kappa*cut*cut/4-.5*min(dr_cap,cut))
    return min(candidates,key=lambda x:(cost(x),x)),cost

def incentive_cases(ref):
    cases=[]
    for supply in [0,4]:
        for cap in [0,4]:
            cases.append(binary_response(ref,ref.Supply(supply,4,dr_cap=cap,dr_price=.5),.5))
    best=min(cases,key=lambda x:-x['reward'])
    assert best['supply']==4 and best['DR_cap']==0
    close(best['reward'],-.25)
    rate_scan=[]
    for rate in [0,.1,.249,.25,.251,.5,2]:
        rows=[binary_response(ref,ref.Supply(s,4,dr_cap=p,dr_price=.5),rate) for s in [0,4] for p in [0,4]]
        withheld=next(x for x in rows if x['supply']==0 and x['DR_cap']==0)
        supplied=next(x for x in rows if x['supply']==4 and x['DR_cap']==0)
        close(withheld['reward'],-rate)
        assert (supplied['reward']>withheld['reward'])==(rate>.25)
        rate_scan.append({'rate':rate,'normal_reward':supplied['reward'],'unpaid_withholding_reward':withheld['reward']})
    value_case=binary_response(ref,ref.Supply(0,4,dr_price=.5,dr_value=.4),.5)
    close(value_case['reward'],-.1)
    high_cost_case=binary_response(ref,ref.Supply(0,4,pg_price=1.7,dr_price=.5,dr_value=.4),2)
    high_cost_normal=binary_response(ref,ref.Supply(4,4,pg_price=1.7,dr_price=.5,dr_value=.4),2)
    close(high_cost_normal['reward'],-1.3);close(high_cost_case['reward'],-1.6)
    old_best=(-math.inf,None);new_best=(-math.inf,None)
    for si in range(101):
        for pi in range(101):
            supply,cap=si*.04,pi*.04
            cut,_=continuous_response(supply,cap)
            orders=[ref.Order('A',4,curtail=cut,uc_tariff=.4)]
            s=ref.Supply(supply,4,dr_cap=cap,dr_price=.5)
            r=ref.run(orders,s);n=revised_result(orders,s,r,.5)
            if r['cash']>old_best[0]:old_best=(r['cash'],[supply,cap,cut])
            if n['reward']>new_best[0]:new_best=(n['reward'],[supply,cap,cut])
    close(old_best[0],0);close(new_best[0],-.15);assert new_best[1][:2]==[4.0,0.0]
    # Fixed response shows compensation relabeling; not a follower-equilibrium claim.
    o=[ref.Order('A',4,curtail=4,uc_tariff=.4)]
    a=[]
    for cap in [0,4]:
        s=ref.Supply(0,4,dr_cap=cap,dr_price=.5);r=ref.run(o,s)
        a.append(revised_result(o,s,r,2))
    close(a[0]['reward'],-2);close(a[1]['reward'],-.5)
    # Feasible local bounds witness: cut 4, shift -2, due .5 kWh, flex 6.
    b=ref.local_bounds(4,0,6,.5,.5,2,.25,0,None,8,1)
    close(b['curtail'],4);close(b['low'],-2);close(b['high'],-2)
    o=[ref.Order('A',4,curtail=4,repay=2,due_attempt=2,uc_tariff=.4)]
    s=ref.Supply(2,4,dr_cap=2,dr_price=.5);r=ref.run(o,s);extra=revised_result(o,s,r,2)
    close(r['consumers']['A']['native_lost'],0);close(r['consumers']['A']['repayment'],2)
    close(extra['capacity_gap_kw'],2);close(extra['capacity_assessment'],1)
    # Free capacity is not actual import; independent firm charging must still execute.
    o=[ref.Order('A',4,curtail=2,uc_tariff=.4)];flat=[]
    for offer in [4,8]:
        s=ref.Supply(offer,8);r=ref.run(o,s);flat.append(revised_result(o,s,r))
        close(r['pg'],2)
    close(flat[0]['cash'],flat[1]['cash']);close(flat[0]['reward'],flat[1]['reward'])
    for event,price,expected in [(True,0,0),(False,.5,0),(True,.5,4)]:
        credit,gap,cost=assessment(4,0,0,4,4,event,price,.25,2)
        close(credit,expected)
    natural=assessment(4,0,4,0,2,True,.5,.25,2);close(natural[2],0)
    return {'original_binary_U03_rechecked':cases,'rate_scan':rate_scan,'positive_value_can_restore_advantage':value_case,
            'bounded_high_cost_value_comparison':{'supplied':high_cost_normal,'withheld':high_cost_case},
            'continuous_response_grid':{'leader_candidates':10201,'kappa':.5,'old_best':old_best,'new_best':new_best,
                                        'scope':'Analytical scalar consumer response, finite leader grid, single-period no-DER/ESS/queue case'},
            'A01_existing_response_purchased':a,'A02_due_extra_reservation':extra,'A03_capacity_flat_region':flat,
            'positive_price_and_event_gating':'passed','adequate_capacity_unpaid_voluntary_DR_cost':natural[2]}

def legal_supply(ref,rng,energy,dt=.25):
    der=rng.uniform(0,10);gridmax=rng.uniform(1,12);grid=rng.uniform(0,gridmax)
    curtailed=rng.uniform(0,der);room=min(8,(10-energy)/(.95*dt))
    cr=cg=dis=0.0
    sample=rng.randrange(3)
    if sample==0:
        cr=rng.uniform(0,min(der-curtailed,room));cg=rng.uniform(0,min(grid,room-cr))
    elif sample==1:dis=rng.uniform(0,min(8,.95*energy/dt))
    event=bool(rng.randrange(2))
    return ref.Supply(grid,gridmax,der=der,planned_curtail=curtailed,der_charge=cr,grid_charge=cg,
                      discharge_offer=dis,dr_cap=rng.uniform(0,8) if event else 0,energy=energy,dt=dt,event=event,
                      pg_price=rng.uniform(.2,1.7),dr_price=rng.choice([0,.2,.5]) if event else 0,dr_value=rng.choice([0,.4]),
                      degradation=.02,curtail_opportunity=.1)

def physical_and_assessment_checks(ref):
    rng=random.Random(SEED);maximum_power=maximum_cash=maximum_queue=0.0
    for _ in range(5000):
        dt=rng.choice([.25,.5,1]);s=legal_supply(ref,rng,rng.uniform(0,10),dt)
        orders=[]
        for i in range(rng.randint(1,8)):
            base=rng.uniform(0,10);cut=rng.uniform(0,base);defer=rng.uniform(0,base-cut);repay=rng.uniform(0,4) if defer==0 else 0
            orders.append(ref.Order(str(i),base,cut,defer,repay,der_share=rng.random(),der_bid=.3,uc_tariff=.9))
        immutable=asdict(s);r=ref.run(orders,s);n=revised_result(orders,s,r)
        assert asdict(s)==immutable
        maximum_power=max(maximum_power,abs(r['power_residual']));maximum_cash=max(maximum_cash,abs(r['cash_residual']))
        # Independent quantity interpretation plus monotonicity; actual payment remains the original one.
        raw=sum(o.baseline+o.due_attempt for o in orders)-(s.der-s.planned_curtail-s.der_charge+s.grid_offer+s.discharge_offer-s.grid_charge)
        credit=r['dr_payment']/(s.dr_price*s.dt) if s.event and s.dr_price>0 else 0
        close(n['credited_DR_kw'],credit);close(n['capacity_assessment'],2*max(0,raw-credit)*dt)
        close(n['reward'],r['cash']-r['degradation_cost']-r['curtail_cost']+r['benefit']-n['service_cost'])
        assert assessment(n['baseline_kw'],n['due_kw'],n['announced_capacity_kw']+1,s.dr_cap,r['net_dr'],s.event,s.dr_price,dt)[2]<=n['capacity_assessment']+1e-9
    for episode in range(100):
        energy=5.0;queues=[[] for _ in range(4)]
        for t in range(96):
            s=legal_supply(ref,rng,energy);orders=[]
            before_backlog=[]
            for i,q in enumerate(queues):
                base=rng.uniform(1,8);rigid=base*rng.uniform(0,.5);flex=rng.uniform(0,6)
                backlog=sum(x[1] for x in q);due=sum(e for d,e in q if d<=t)
                b=ref.local_bounds(base,rigid,flex,backlog,due,3,.25,t,None,8,rng.random())
                shift=rng.uniform(b['low'],b['high']);before_backlog.append(backlog)
                orders.append(ref.Order(str(i),base,b['curtail'],max(0,shift),max(0,-shift),b['attempt'],rng.random(),.3,.9))
            original_action=asdict(s);r=ref.run(orders,s);late=0
            for i,o in enumerate(orders):
                c=r['consumers'][str(i)]
                queues[i]=ref.apply_queue(queues[i],t,8,o.defer,c['repayment'],.25)
                backlog=sum(e for _,e in queues[i]);expected=before_backlog[i]+(o.defer-c['repayment'])*.25
                maximum_queue=max(maximum_queue,abs(backlog-expected));assert -1e-9<=backlog<=3+1e-9
                late+=sum(e for d,e in queues[i] if d<=t)
            n=revised_result(orders,s,r,late_energy=late)
            assert n['service_cost']>=n['capacity_assessment']-1e-9
            assert asdict(s)==original_action;energy=r['energy_next']
            maximum_power=max(maximum_power,abs(r['power_residual']));maximum_cash=max(maximum_cash,abs(r['cash_residual']))
    # Equivalent reward expansion removes an intermediate book-profit dependency.
    for _ in range(10000):
        cash,ess,spill,value,cost=[rng.uniform(-5,5) for i in range(5)]
        book=cash-ess-spill;close(book+value-cost,cash-ess-spill+value-cost)
    return {'static_samples':5000,'continuous_episodes':100,'physical_periods':9600,'consumer_periods':38400,
            'reward_expansion_equivalence_samples':10000,'maximum_power_residual_kw':maximum_power,
            'maximum_cash_residual_yuan':maximum_cash,'maximum_queue_residual_kwh':maximum_queue,
            'original_actions_and_firm_execution_preserved':True}

def units_observations_and_worked_example(ref):
    units=[]
    for dt in [.125,.25,.5,1]:
        credit,gap,cost=assessment(4,0,0,0,4,True,.5,dt,2)
        close(cost,8*dt);units.append({'dt_h':dt,'cost':cost})
    # W/Wh conversion: price and assessment rates become per Wh.
    close(assessment(4,0,0,0,4,True,.5,.25,2)[2],assessment(4000,0,0,0,4000,True,.0005,.25,.002)[2])
    # A heterogeneous capability witness prevents an invalid min-of-sums simplification.
    local_due=sum(min(d/.25,f) for d,f in [(1,0),(0,4)])
    aggregated_wrong=min(sum([1,0])/.25,sum([0,4]));close(local_due,0);close(aggregated_wrong,4)
    H=8;obs=[0.0]*13+[0.0]*(H+1)+[0.0]*6;assert len(obs)==H+20
    orders=[ref.Order('A',6,1,1,der_share=.5,der_bid=.7),ref.Order('B',5,1,der_share=.5,der_bid=.65),ref.Order('C',4,repay=1,der_share=.4,der_bid=.6)]
    s=ref.Supply(6,8,der=6,planned_curtail=1,der_charge=1,grid_charge=1,dr_cap=2,energy=5,
                 dr_value=.4,degradation=.02,curtail_opportunity=.1)
    result=ref.run(orders,s);current=revised_result(orders,s,result)
    close(current['cash'],-.025);close(current['capacity_assessment'],2);close(current['reward'],-1.86)
    expanded_supply=ref.Supply(8,8,der=6,dr_cap=2,energy=5,dr_value=.4,degradation=.02,curtail_opportunity=.1)
    expanded=ref.run(orders,expanded_supply);expanded_current=revised_result(orders,expanded_supply,expanded)
    close(expanded_current['reward'],1.5125)
    return {'time_factor_checks':units,'W_Wh_conversion':'passed','UC_observation_dimension':H+20,
            'heterogeneous_due_counterexample':{'sum_local_due_kw':local_due,'invalid_min_of_aggregates_kw':aggregated_wrong},
            'worked_example':current,'expanded_supply_example':expanded_current}

def main():
    ref=load_reference();document=check_document();incentives=incentive_cases(ref)
    physics=physical_and_assessment_checks(ref);units=units_observations_and_worked_example(ref)
    integrity={}
    for name in ['protected_files_manifest.json','history_files_manifest.json']:
        manifest=json.loads((BASE/name).read_text(encoding='utf-8'))
        changed=[path for path,value in manifest.items() if sha(ROOT/path)!=value]
        assert not changed,changed
        integrity[name]={'files':len(manifest),'unchanged':True}
    report={'status':'passed','date':'2026-10-14','seed':SEED,'source_before_sha256':json.loads((BASE/'u03_revision_baseline.json').read_text(encoding='utf-8'))['source_before_sha256'],
            'source_after_sha256':sha(BASE/'modeling_complete_zh.md'),'document':document,'incentives':incentives,
            'physics_cash_and_queue':physics,'units_and_observations':units,'integrity':integrity,
            'production_imports':False,'reference_verifier_unchanged':True,
            'remaining_findings':['A01 compensation can purchase existing response without additionality','A02 due reference yields conservative extra reservation','A03 free announcement capacity produces flat regions','A04 coefficient/value/physical-resource/baseline/dynamic conditions'],
            'scope':'Independent documentation reference, current reward arithmetic, physical/cash invariants and stated response candidates. Production environment, algorithms and tests unchanged; general equilibrium and learning performance require separate experiments.'}
    (BASE/'u03_revision_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'status':'passed','document':document,'physics':physics,'integrity':integrity},ensure_ascii=False))

if __name__=='__main__':main()
