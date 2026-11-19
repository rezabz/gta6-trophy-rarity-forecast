"""gta6-trophy-rarity-forecast-01 entry: cohort / dilution forecaster.

Model (see BUILD-PLAN.md): the earn rate is holders over tracked players,
r_i(t) = E_i(t) / N(t). Forecast the two counts separately, not the rate.

  1. Collapse consecutive identical reads into distinct states (Exophase's
     page is frozen between refreshes). Each state: time, N, holder counts
     E_i = round(rate_i * N / 100).
  2. Forecast N(target):
     - >= 2 distinct states: OLS fit of ln N on ln(hours since release),
       slope clamped to [0.05, 1.2], read at the target.
     - 1 distinct state (flat history): the page has been frozen F hours and
       refreshes come roughly every 4-6 h early on, so the next refresh is
       due within (6 - F) h; P(refresh before target) = horizon / (6 - F)
       (capped at 1). The first refresh multiplies N ~6x (the launch prior
       from the brief: 638 -> 3868 tracked players); later refreshes are
       assumed to double N, with growth slowing.
  3. Forecast E_i(target) by cohort decomposition:
       E_i(T) = E_i(last) + (N(T) - N_last) * s_i
     where s_i is the share of newly tracked players holding trophy i,
     estimated from observed refreshes (recency-weighted mean of
     (E_ik - E_i,k-1) / (N_k - N_{k-1}), shrunk toward a difficulty prior)
     or, when no refresh was ever observed, from the difficulty prior alone.
     A progression term (power law of E_i in time, shrunk toward the
     cross-trophy median exponent) is combined by taking the max: holders
     never decrease and either mechanism can drive growth.
  4. Median rate m_i = E_i(T) / N(T). A trophy last at >= 99.99% (the
     intro activity every player completes) forecasts all quantiles 100.
  5. Spread on the logit scale:
       sigma = sqrt(0.15^2 + (0.35*sqrt(horizon_days))^2 + sigma_refresh^2)
     with sigma_refresh = 0.5 when no refresh was ever observed (its size
     is unknown) and 0.25 once refreshes are in the history.
     q_tau = inv_logit(logit(m_i) + z_tau * sigma), rounded half-up to
     4 decimals.

Stdlib only, no network, deterministic. Reads every *.in in
MAKEFUN_INPUT_DIR (default /input), writes <id>.out to MAKEFUN_OUTPUT_DIR
(default /output).
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]

# --- model constants (structure / priors, not tuned to any case answer) ---
SIGMA0 = 0.15            # read noise, logit units
SIGMA_TREND = 0.35       # trend uncertainty per sqrt(day)
SIGMA_REFRESH_FLAT = 0.5   # refresh size unknown (flat history)
SIGMA_REFRESH_RICH = 0.25  # refreshes observed; future sizes uncertain
REFRESH_INTERVAL_LO = 4.0  # hours; refreshes come every ~4-6 h early on
REFRESH_INTERVAL_HI = 6.0
REFRESH_INTERVAL_MEAN = 5.0
FIRST_REFRESH_N_FACTOR = 6.06  # launch prior: first refresh 638 -> 3868
LATER_REFRESH_N_FACTOR = 2.0   # later refreshes multiply N by less
PRIOR_PSEUDO_WEIGHT = 1.0      # shrinkage weight of the difficulty prior
PROG_SHRINK = 0.5              # weight of a trophy's own progression slope
MIN_T_HOURS = 0.25             # floor for hours-since-release (log scale)


def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def clamp_pct(x):
    return clamp(x, 0.01, 99.99)


def logit(pct):
    p = clamp_pct(pct) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit_pct(x):
    if x >= 0:
        return 100.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (like the starter / JavaScript)."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def median(xs):
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return 0.0
    if n % 2:
        return ys[n // 2]
    return 0.5 * (ys[n // 2 - 1] + ys[n // 2])


def ols_slope_intercept(xs, ys):
    """Least squares y = a + b*x; returns (a, b)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0:
        return my, 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return my - b * mx, b


# ---------------------------------------------------------------- priors

def difficulty_prior(trophy):
    """Share of a newly tracked cohort expected to hold this trophy.

    From trophy difficulty (tier + what the trophy asks for), calibrated to
    the one refresh the brief itself publishes (638 -> 3868 players, only
    ~103 new holders outside the intro trophy): the intro activity is held
    by everyone, the first chapter by a few per thousand, a fishing
    challenge by about one in a hundred, arcade / vehicle trophies by a few
    per thousand, other story missions by well under one per thousand, and
    most other trophies by a fraction of that.
    """
    text = (trophy.get("name", "") + " " + trophy.get("description", "")).lower()
    tier = trophy.get("tier", "bronze")
    if "gone fishing" in text:
        return 1.0
    if "chapter 1" in text:
        return 0.003
    if "fishing" in text:
        return 0.01
    if "arcade" in text:
        return 0.003
    if "vehicle" in text:
        return 0.002
    if text.strip().startswith("complete"):
        return 0.0006
    if tier == "platinum":
        return 0.00005
    if tier == "gold":
        return 0.0001
    if tier == "silver":
        return 0.0002
    return 0.0003


# ---------------------------------------------------------------- states

def distinct_states(case):
    """Collapse consecutive identical reads into distinct states.

    Returns list of dicts {t, n, e} with t = hours since release (floored),
    n = players tracked, e = [holder counts] (rounded from rates).
    """
    release = parse_time(case["game"]["release_date"] + "T00:00:00Z") \
        if len(case["game"].get("release_date", "")) == 10 else \
        parse_time(case["game"]["release_date"])
    states = []
    prev_key = None
    for read in case["history"]:
        rates = read["rates"]
        key = (read["players_tracked"], tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        n = read["players_tracked"]
        at = parse_time(read["at"])
        t = max(MIN_T_HOURS, (at - release).total_seconds() / 3600.0)
        e = []
        for r in rates:
            e.append(0 if r is None else int(round(r * n / 100.0)))
        states.append({"t": t, "n": n, "e": e, "at": at})
    return states


# ---------------------------------------------------------------- forecast

def forecast_n(states, horizon_hours, checkpoint, first_read_at):
    """Forecast players tracked at the target."""
    n_last = states[-1]["n"]
    if len(states) >= 2:
        xs = [math.log(s["t"]) for s in states]
        ys = [math.log(s["n"]) for s in states]
        a, b = ols_slope_intercept(xs, ys)
        b = clamp(b, 0.05, 1.2)
        t_target = states[-1]["t"] + horizon_hours
        n_t = math.exp(a + b * math.log(max(MIN_T_HOURS, t_target)))
        return clamp(n_t, n_last, n_last * 30.0)
    # Flat history: no refresh ever observed. The next refresh is due
    # within (REFRESH_INTERVAL_HI - F) hours, F = hours frozen so far.
    frozen = max(0.0, (checkpoint - first_read_at).total_seconds() / 3600.0)
    window = max(0.5, REFRESH_INTERVAL_HI - frozen)
    p_first = clamp(horizon_hours / window, 0.0, 1.0)
    extra = max(0.0, (horizon_hours - window) / REFRESH_INTERVAL_MEAN)
    n_t = n_last * (1.0 + p_first * (FIRST_REFRESH_N_FACTOR - 1.0))
    n_t *= LATER_REFRESH_N_FACTOR ** extra
    return clamp(n_t, n_last, n_last * 50.0)


def cohort_rates(states, trophies):
    """Per-trophy share s_i of new arrivals holding the trophy."""
    n_tr = len(trophies)
    priors = [difficulty_prior(t) for t in trophies]
    num = [0.0] * n_tr
    den = [0.0] * n_tr
    # Consecutive distinct-state pairs are the observed refreshes.
    for k in range(1, len(states)):
        prev, cur = states[k - 1], states[k]
        dn = cur["n"] - prev["n"]
        if dn <= 0:
            continue
        w = float(k)  # later refreshes count more: arrivals played longer
        for i in range(n_tr):
            s_ik = clamp((cur["e"][i] - prev["e"][i]) / dn, 0.0, 1.0)
            num[i] += w * s_ik
            den[i] += w
    out = []
    for i in range(n_tr):
        if den[i] > 0:
            out.append((num[i] + PRIOR_PSEUDO_WEIGHT * priors[i])
                       / (den[i] + PRIOR_PSEUDO_WEIGHT))
        else:
            out.append(priors[i])
    return out


def progression_exponents(states, trophies):
    """Shrunk power-law exponents of holder counts vs time, per trophy."""
    n_tr = len(trophies)
    betas = [None] * n_tr
    if len(states) >= 3:
        for i in range(n_tr):
            pts = [(math.log(s["t"]), math.log(s["e"][i]))
                   for s in states if s["e"][i] > 0]
            if len(pts) >= 3:
                _, b = ols_slope_intercept([p[0] for p in pts],
                                           [p[1] for p in pts])
                betas[i] = clamp(b, 0.0, 1.5)
    fitted = [b for b in betas if b is not None]
    beta_med = median(fitted) if fitted else 0.0
    return [beta_med if b is None else
            clamp(PROG_SHRINK * b + (1.0 - PROG_SHRINK) * beta_med, 0.0, 1.5)
            for b in betas], beta_med


def forecast_case(case, name):
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")
    trophies = case["trophies"]
    n_tr = len(trophies)
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    checkpoint = parse_time(case["at"])
    first_read_at = parse_time(case["history"][0]["at"])
    last_rates = case["history"][-1]["rates"]

    states = distinct_states(case)
    n_target = forecast_n(states, horizon_hours, checkpoint, first_read_at)
    n_last = states[-1]["n"]
    e_last = states[-1]["e"]
    s = cohort_rates(states, trophies)
    betas, _ = progression_exponents(states, trophies)
    t_last = states[-1]["t"]
    t_target = t_last + horizon_hours

    sigma_refresh = SIGMA_REFRESH_FLAT if len(states) < 2 else SIGMA_REFRESH_RICH
    sigma = math.sqrt(SIGMA0 ** 2 + (SIGMA_TREND * math.sqrt(horizon_days)) ** 2
                      + sigma_refresh ** 2)

    rows = []
    for i, trophy in enumerate(trophies):
        last = last_rates[i]
        # Intro trophy every tracked player holds: stays 100%.
        if last is not None and last >= 99.99:
            rows.append({"id": trophy["id"], "q": [100, 100, 100, 100, 100]})
            continue
        e_cohort = e_last[i] + (n_target - n_last) * s[i]
        if len(states) >= 3 and e_last[i] > 0:
            e_prog = e_last[i] * (t_target / t_last) ** betas[i]
        else:
            e_prog = float(e_last[i])
        e_t = clamp(max(e_cohort, e_prog), float(e_last[i]), float(n_target))
        m = clamp_pct(100.0 * e_t / n_target)
        lg = logit(m)
        qs = [round4(clamp(inv_logit_pct(lg + z * sigma), 0.0, 100.0))
              for z in Z]
        # Rounding can only tie, never invert; enforce non-decreasing.
        for j in range(1, 5):
            if qs[j] < qs[j - 1]:
                qs[j] = qs[j - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
