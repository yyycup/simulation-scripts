import pandas as pd
from pathlib import Path
from thermal_batch_config import AMBIENT_TEMP_K
from thermal_loop import staged_fan_speed
from thermal_system import pump_model, run_refrigeration_cycle

rows=[]
for n in (2000.,3000.,4000.,5000.,6000.):
 for p in (1600.,2400.,3200.,4000.,4800.):
  m,_=pump_model(p)
  for t in (20.,25.,30.,35.):
   r=run_refrigeration_cycle(n,staged_fan_speed(n),t+273.15,m,AMBIENT_TEMP_K)
   rows.append({'N_comp_rpm':n,'N_pump_rpm':p,'T_cool_in_C':t,'Q_evap_plant_W':r['Q_evap'],'Q_hx_potential_W':r['Q_hx_potential'],'Q_ref_max_W':r['Q_ref_max'],'limit_type':'Q_hx' if r['Q_hx_potential']<=r['Q_ref_max'] else 'Q_ref_max'})
out=Path('outputs')/'mpc_evaporator_capacity_candidate_b'/'candidate_b_capacity_limits_grid.csv'
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
print(out)
