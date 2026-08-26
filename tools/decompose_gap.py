"""
decompose_gap.py -- split a CVRP solution's total cost into "depot legs" vs
"inter-customer legs", so a cost gap between two solvers can be attributed to
route STRUCTURE (how many vehicles / how much depot round-tripping) versus tour
QUALITY (how well customers are sequenced within a route).

Motivation: report 010 establishes we lose to FILO2 by ~1.09% on Valle-D'Aosta
despite deploying ~2x its total search work. This tool answers *where* that
1.09% physically sits in the solution.

Reuses tools/score_sol.py's verified .vrp parsing and id-mapping (see its module
docstring -- the id convention is subtle and was empirically validated there).
Follows the same "recompute, don't trust the reported number" discipline as
src/verifier.py: every decomposition is only reported if the recomputed total
matches the cost the file itself claims.

Usage:
    python tools/decompose_gap.py <instance.vrp> ours=<our_solution.txt> filo2=<filo2.sol>
"""
import re
import sys

from score_sol import parse_vrp, parse_sol, dist


def routes_from_our_format(path, bks_to_file, depot_id):
    """Our solver writes 'Route N (Load: L): 0 -> a -> b -> 0' using internal ids
    (0 = depot, 1..n = customers in the same renumbered order score_sol.py maps),
    plus a 'Final Cost:' header line."""
    routes = []
    reported = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            m = re.match(r"Final Cost:\s*(\d+)", line)
            if m:
                reported = int(m.group(1))
                continue
            m = re.match(r"Route\s+\d+\s*\([^)]*\):\s*(.*)", line)
            if not m:
                continue
            toks = [t.strip() for t in m.group(1).split("->")]
            ids = [int(t) for t in toks if t]
            # strip the leading/trailing depot sentinels (internal id 0)
            cust = [bks_to_file[i] for i in ids if i != 0]
            if cust:
                routes.append(cust)
    return routes, reported


def routes_from_filo2_sol(path, bks_to_file):
    raw, embedded = parse_sol(path)
    return [[bks_to_file[i] for i in r] for r in raw], embedded


def decompose(coords, depot_id, routes):
    depot_legs = 0
    inter = 0
    customers = 0
    for r in routes:
        depot_legs += dist(coords, depot_id, r[0])
        depot_legs += dist(coords, r[-1], depot_id)
        for a, b in zip(r, r[1:]):
            inter += dist(coords, a, b)
        customers += len(r)
    return {
        "routes": len(routes),
        "customers": customers,
        "depot_legs": depot_legs,
        "inter_customer": inter,
        "total": depot_legs + inter,
    }


def main():
    vrp = sys.argv[1]
    args = dict(a.split("=", 1) for a in sys.argv[2:])
    coords, depot_id, bks_to_file = parse_vrp(vrp)

    results = {}
    if "ours" in args:
        routes, reported = routes_from_our_format(args["ours"], bks_to_file, depot_id)
        d = decompose(coords, depot_id, routes)
        d["reported"] = reported
        results["ours"] = d
    if "filo2" in args:
        routes, embedded = routes_from_filo2_sol(args["filo2"], bks_to_file)
        d = decompose(coords, depot_id, routes)
        d["reported"] = embedded
        results["filo2"] = d

    for name, d in results.items():
        ok = "OK" if d["reported"] is None or abs(d["total"] - d["reported"]) <= 1 \
             else f"MISMATCH (file says {d['reported']})"
        print(f"=== {name} === [recompute check: {ok}]")
        print(f"  routes                 {d['routes']:,}")
        print(f"  customers served       {d['customers']:,}")
        print(f"  depot-leg cost         {d['depot_legs']:,}  "
              f"({100.0*d['depot_legs']/d['total']:.2f}% of total)")
        print(f"  inter-customer cost    {d['inter_customer']:,}  "
              f"({100.0*d['inter_customer']/d['total']:.2f}% of total)")
        print(f"  total                  {d['total']:,}")
        print()

    if len(results) == 2:
        o, f = results["ours"], results["filo2"]
        gap = o["total"] - f["total"]
        dl = o["depot_legs"] - f["depot_legs"]
        ic = o["inter_customer"] - f["inter_customer"]
        print("=== gap attribution (ours - filo2) ===")
        print(f"  total gap              {gap:+,}  ({100.0*gap/f['total']:+.3f}%)")
        print(f"    from depot legs      {dl:+,}  ({100.0*dl/gap:.1f}% of the gap)")
        print(f"    from inter-customer  {ic:+,}  ({100.0*ic/gap:.1f}% of the gap)")
        print(f"  extra routes           {o['routes'] - f['routes']:+,}")
        if o["routes"] != f["routes"]:
            print(f"  cost per extra route   {dl/(o['routes']-f['routes']):,.0f} "
                  f"(depot-leg delta / extra routes)")


if __name__ == "__main__":
    main()
