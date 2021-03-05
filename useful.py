za =[(13,6),(13,8),(8,10),(3,14),(4,16),(2,16)]
D, E, F, G, H, I = za
ne_eps = lambda p, eps: [x for x in l1 if distf(x[1], p) <= eps]
distf = lambda p1, p2: math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
l1 = [("A", A), ("B", B), ("C", C), ("D", D), ("E", E), ("F", F), ("G", G), ("H", H), ("I", I)]
is_in_ne = lambda p,q, eps: True if len([x for x in ne_eps(q, eps) if p == x[1]]) > 0 else False
is_dr = lambda p,q,eps,minpts: True if is_in_ne(p,q,eps) and len(ne_eps(q,eps)) >= minpts else False