import numpy as np
avg_log_sqft = np.log(1800)
avg_log_lot  = np.log(5800)
avg_beds = 3.2; avg_baths = 1.0 + 0.0012*1800
avg_age  = 53;  avg_grade = 7.2; avg_cond = 3.2
avg_garage = 0.68; avg_renov = 0.18; avg_zip_prem = 0.153

fs = (0.80*avg_log_sqft + 0.03*avg_beds + 0.09*avg_baths + 0.10*avg_log_lot
      - 0.004*avg_age + 5e-6*avg_age**2 + 0.09*avg_grade + 0.05*avg_cond
      + 0.06*avg_garage + 0.05*avg_renov + avg_zip_prem)

b0 = np.log(650_000) - fs
print(f"Feature sum  : {fs:.4f}")
print(f"Intercept b0 : {b0:.4f}")
print(f"Sanity median: ${np.exp(b0+fs):,.0f}")
