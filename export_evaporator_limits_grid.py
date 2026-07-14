from pathlib import Path
import pandas as pd
from run_mpc_evaporator_capacity_calibration import N_COMP,N_PUMP,T_COOL,plant_point

rows=[]
for n in N_COMP:
 for p in N_PUMP:
  for t in T_COOL:
   q,limit=plant_point(n,p,t)
   rows.append({'N_comp_rpm':n,'N_pump_rpm':p,'T_cool_in_C':t,'Q_evap_plant_W':q,'limit_type':limit})
out=Path('outputs')/'mpc_evaporator_capacity_candidate_b'/'candidate_b_capacity_grid_with_limits.csv'
pd.DataFrame(rows).to_csv(out,index=False,encoding='utf-8-sig')
print(out)
