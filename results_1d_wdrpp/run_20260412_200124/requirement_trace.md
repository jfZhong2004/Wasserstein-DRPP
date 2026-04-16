# Requirement Trace

1. System C only; steps are fixed to k=0..31 (32 steps).
2. Control modes: zero + PID (LQR excluded this round).
3. PID uses fixed tuned gains and saturation u in [-4,4] with anti-windup.
4. Noise library W1-W6 uses unified variance scale (var=1).
5. W4 uses truncated Student-t(df=3) then variance normalization.
6. Oracle uses true noise distribution only.
7. gamma0(z)=min(0.3||z||_2, 5).
8. Wasserstein radius follows Theorem 3.4 / Eq.(8) with beta=0.05.
9. W-DRPP solver reuses src/solvers/drpp_1d_exact_solver.py directly.
10. Training and evaluation trajectories use different random seed masters (no seed overlap).
