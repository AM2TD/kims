"""UTIL paper (Zhao et al., IJRR 2024, arXiv 2203.14471 v6) Table 4: per-flight 3D position RMSE [m] of the authors' ESKF
(IMU + TDOA), as published. Entries "c q tdoa2 tdoa3": c = constellation, q = trial (const1-3), "t.j" = trial t, trajectory j
(const4), m1-m4 = trial 7 manual flights; "-" = flight not recorded in that mode. T4 maps the data/util flight names to the value."""
_T4 = """c1 1 0.108 0.144|c1 2 0.122 0.144|c1 3 0.109 0.123|c1 4 0.111 0.127|c1 5 0.115 0.149|c1 6 0.123 0.135
c2 1 0.103 0.114|c2 2 0.114 0.116|c2 3 0.108 0.117|c2 4 0.094 0.121|c2 5 0.122 0.125|c2 6 0.106 0.117
c3 1 0.245 0.232|c3 2 0.194 0.225|c3 3 0.189 0.214|c3 4 0.215 0.204|c3 5 0.220 0.241|c3 6 0.193 0.211
c3 m1 0.265 -|c3 m2 0.335 -|c3 m3 - 0.340|c3 m4 - 0.355|c4 1.1 0.201 0.207|c4 1.2 0.168 0.163
c4 1.3 0.194 0.190|c4 2.1 0.231 0.221|c4 2.2 0.200 0.219|c4 2.3 0.222 0.238|c4 3.1 0.919 0.742|c4 3.2 0.718 0.558
c4 3.3 0.651 0.674|c4 4.1 0.818 0.759|c4 4.2 0.766 0.845|c4 4.3 0.735 0.838|c4 5.1 0.473 0.432|c4 5.2 0.470 0.435
c4 5.3 0.395 0.443|c4 6.1 0.509 0.623|c4 6.2 0.459 0.618|c4 6.3 0.517 0.578|c4 m1 0.529 -|c4 m2 0.587 -
c4 m3 0.333 -"""
T4 = {}
for e in _T4.replace("\n", "|").split("|"):
    c, q, e2, e3 = e.split(); c = int(c[1])
    for mode, ev in (("tdoa2", e2), ("tdoa3", e3)):
        if ev == "-": continue
        if q.startswith("m"): name = f"const{c}-trial7-{mode}-manual{q[1:]}"
        elif c < 4: name = f"const{c}-trial{q}-{mode}"
        else: t, j = q.split("."); name = f"const4-trial{t}-{mode}-traj{j}"
        T4[name] = float(ev)
