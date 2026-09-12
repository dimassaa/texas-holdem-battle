# Results: Which No-Bluff Style Survives and Profits?

This is the analysis writeup for the production campaign (10 combinations x 5 seeds x a nominal 50,000 hands, config digest `21f78e32923c9779`). It answers the core question of `docs/Texas Hold'em Simulator for Strategy Analysis (No Bluff).md`: *which style is most profitable in the long run, and how does mixing styles change it?* — under the rules the simulator actually implements. The raw data behind every number is the committed `output/reports/summary_sessions.csv` and `output/reports/summary_combo.csv`.

---

## 1. Setup recap

- 6 seats, blinds 1/2, initial stack 400 chips (= 200 big blinds).
- Pot-limit: raises capped by the pot, `max_aggressions = 4` (bet + 3 raises), all-in as the escape hatch.
- First-passage ruin: a seat whose stack falls below the big blind is frozen; the session ends early when fewer than 2 seats remain.
- **No rake, no rebuys.** Chips are conserved; a busted seat never returns.
- Equity: preflop from a precomputed table, postflop by Monte Carlo at 400 iterations on a Numba fast path.
- No bluffing; the only aggression without a made/strong hand is the documented semi-bluff family (Aggressive c-bets, Adaptive steals, Mathematician EV-justified raises).
- Every session deterministic: identical `seed + config` produces a byte-identical raw log.

The defining property of this regime: because chips never re-enter, total chip flow is **zero-sum by conservation**, and tables die from the inside. "Profitability" therefore collapses into two conflated quantities — how much a style loses, and how long its table stays alive to sample that loss. Everything below is read in that light.

## 2. Headline BB/100

BB/100 = `(net / big_blind) / live_hands * 100`, averaged over the six seats and five seeds of each combination. The 95% CI is `2 * sigma / sqrt(seeds)`. Per-seat values: `output/reports/summary_combo.csv`.

| Combo | BB/100 (mean, 95% CI) | Worst seat mean | Best seat mean |
|---|---|---|---|
| `six_tight`        |  −2.5  ± 2.8     | −6.3   | −0.04 |
| `six_passive`      |  −7.1  ± 8.9     | −16.3  | −0.95 |
| `six_mathematician`| −47.7  ± 58.1    | −114.5 | −5.05 |
| `six_adaptive`     | −50.5  ± 57.4    | −99.3  | −22.75 |
| `benchmark_mix`    | −53.4  ± 56.7    | −104.6 | 16.67 |
| `tight3_agg3`      | −54.7  ± 57.6    | −86.1  | −17.27 |
| `tight3_loose3`    | −58.8  ± 61.3    | −117.3 | −10.71 |
| `six_aggressive`   | −90.1  ± 97.5    | −148.9 | −47.56 |
| `six_loose`        | −173.5 ± 150.3   | −408.2 | −4.57 |
| `agg2_passive4`    | −220.6 ± 241.7   | −539.5 | −84.21 |

Two facts deserve emphasis before the ranking is interpreted.

First, **every** combination is negative on average. In a zero-sum, no-rake, no-rebuy game this is a theorem, not a surprise: whatever one seat wins, the rest lose. A negative mean is not the finding — the *ordering and the spread* are.

Second, the BB/100 spread is almost entirely a **table-longevity** effect, not a skill effect. The best style (Tight, −2.5) is the least-staking style and loses only 2.5 BB per 100 live hands; the worst (the 2-aggressive/4-passive mix, −220.6) is a mix whose own aggressive seats torch the stack before 100 hands of data exist. Note that `benchmark_mix` — the "one of each" table — has a *positive* seat (16.67 BB/100) yet still averages −53.4: some seats win while the table burns around them. That is the survivor-selection signature discussed in section 6.

## 3. Ruin analysis

The per-seat median first-ruin hand (from `summary_combo.csv`, `first_ruin_hand_X_med`) and the per-combo table-death statistics tell the same story more starkly than the BB/100 column:

| Combo | Seat hardest hit (median first-ruin hand) | Median effective hands | Min hands | Sessions truncated (of 5) |
|---|---|---|---|---|
| `six_tight`        | seat 3 (1,154)     | 50,000 | 22,578 | 2 |
| `six_passive`      | seat 3 (1,334)     | 50,000 | 15,527 | 2 |
| `agg2_passive4`    | seat 0 (31)        | 10,738 | 286     | 3 |
| `six_loose`        | seat 0 (28)        | 10,048 | 1,633   | 5 |
| `six_aggressive`   | seat 0 (191)       | 8,375  | 1,024   | 5 |
| `six_adaptive`     | seat 0 (325)       | 5,008  | 3,119   | 5 |
| `benchmark_mix`    | seat 0 (1,035)     | 2,896  | 641     | 5 |
| `tight3_loose3`    | seat 5 (41)        | 2,833  | 1,342   | 5 |
| `tight3_agg3`      | seat 5 (72)        | 2,609  | 1,716   | 5 |
| `six_mathematician`| seat 0 (260)       | 1,144  | 512     | 5 |

(Formulas: median effective hands = median of `num_hands_run` across a combination's 5 sessions; truncated = sessions that ended by table death before 50,000 hands.)

Average ruined seats per 6-seat table by hand 50,000 (campaign data): 5.0 for Loose, Aggressive, Adaptive, Mathematician, `tight3_agg3`, `benchmark_mix`, and `tight3_loose3`; 4.6 for `agg2_passive4`; 4.4 for Tight and Passive.

Interpretation of the ruin curves (`ruin_curve_*.png`):

- **Tight and Passive never really leave the stack-losing business; they leave it slowly.** Median first-ruin hands run into the thousands (Tight seat 2: 13,680; Passive seat 4: 15,526), and half the sessions survive the full 50k. The curves decay gently.
- **The bigger-betting styles die from the first thousand hands.** Loose seat 0/2 median first-ruin at hand 28; the passive-heavy `agg2_passive4` busts seat 0 at hand 31. The `ruin_curve` family drops vertically and the table dies with it.
- **Mathematician is the cleanest failure of the experiment design.** EV-justified pushes and thin-odds calls make it the *highest*-variance style at 400 Monte Carlo iterations, and its median 1,144-hand table produces almost no long-run data. Mathematically-correct betting is indistinguishable in this record from ruin with correct odds.

The single most transferable finding: **the tightest style survives; the mathematically EV-pushing style collapses.** Under first-passage ruin, the optimal strategy is to have the smallest variance, not the highest expectation — and the no-bluff, no-rebuy, pot-limited rules strip away every lever (bluffing, deep-stack buying power, rebuy discipline) that would make high-variance play survivable.

## 4. Starting-hand findings

The committed artifacts include 13x13 starting-hand profitability heatmaps (`output/reports/plots/heatmap_<style>.png`). Because the heatmaps are image-only, the per-cell numbers were re-derived from the raw logs (six_<style>/seed101, first 40,000 hands) to give the findings a measurement rather than a pixel-read basis. Cell values are mean BB per dealt hand; the tight Pure table's 40k-hand cell samples are meaningful, while the six_mathematician table died at ~1.1k hands and its per-cell means sit on single-digit samples.

| Style | Top cells (mean BB/dealt hand) | Bottom cells |
|---|---|---|
| Tight | AA +1.9, KK +1.1, AKs +0.9, JJ/QQ +0.9 | never worse than −0.2 (69s, 44s, 46s) |
| Loose | 36s +3.7, KK +2.6, A5s +2.5, AA +2.4 | −1.5 … −0.7 (68s, A8s, 89s, 66s) |
| Mathematician | AJs +12.6, AA +8.3, AQs +7.8, TT +6.7 | −6.6 … −5.0 (69s, 74s, QTs, 32s) |

- **Tight concentrates profit in premiums, as documented.** The positive region is the 77+ / AQo+ / AJs+ zone and its downside cells are tiny (never below −0.2 BB/hand) — a style whose mean BB/100 (−2.5) is that close to zero *because* the losing cells are that shallow.
- **Loose's spread is real, and its cost is the bottom third.** The ~40% range does score with speculative holdings (36s +3.7) — but the same row of the table produces the −1.5-class cells that are played over and over to showdown, exactly the "plays too much, loses too deep" signature behind its −173.5 BB/100.
- **Mathematician's grid is dominated by sampling noise, not by EV.** With a ~1.1k-hand table and 169 cells, the +12.6 / −6.6 extremes are single-digit hand samples, not per-cell expectation. Nothing about the EV rule concentrates that; it is the direct statistical consequence of the style ruining its own table within the first thousand hands (section 3).

## 5. Positional findings

Positional EV is computed by design: `compute_session_stats` records per-seat, per-zone EV (`early`, `middle`, `late`, `blinds`; button = late, +1/+2 = blinds, +3/+4 = early, rest = middle) and persists it in each session's `session_stats.json`. The committed *report* CSVs, however, only carry per-seat BB/100 and winrate.

What the committed CSVs do support:

- **A persistent seat asymmetry exists.** In `summary_combo.csv`, seat 0 is consistently the worst seat in the fast-busting combinations (six_loose −408 vs. seat 4's −4.6; six_aggressive −148.9; agg2_passive4 −539.5; six_tight −0.7, the mildest of the spread). `summary_sessions.csv` shows the same per-session: ruin hits seat 0 first and hardest.
- **The asymmetry is not a positional-EV effect.** The button rotates every hand, so over a long session every seat sees every position equally. The seat-0 pattern in the wrecking combinations is a *cascade* artifact: when one seat busts early, the remaining seats re-seat relative to the button, and the fixed seat indices inherit unequal blinding/betting pressure for the rest of the session. In other words, seat number is a poor proxy for position once the table is short-handed.
- The clean positional question — *does late position pay more than the blinds for each style?* — is not answerable from the committed reports. It is one aggregation step away (rolling `positional_ev` up to zone x combo from the fifty `session_stats.json` files) and is the cheap, high-value next analysis.

## 6. Limitations and threats to validity

1. **Effective-hand-count falloff (the big one).** Nominal sessions are 50,000 hands; median *effective* sessions range from 50,000 (Tight, Passive) down to 1,144 (Mathematician). A BB/100 estimated on 1,144 live hands across a dying table is a sample of a transient, not an estimate of a stationary quantity. The BB/100 rows for slower tables are statistically fragile by construction.
2. **Survivor-selection bias.** Under first-passage ruin, the BB/100 mean is dominated by the seats that survive — the ones that happened not to bust early. Early-busting seats contribute a large negative net but few live hands (which is exactly why they don't drag the denominator up). The headline mean therefore understates the per-seat loss rate of the fast-busting styles and treats "lived long" as if it were "won more". For `benchmark_mix`, the positive-best-seat row is this bias in its cleanest form.
3. **Cross-seed CI vs. cross-seat variance.** The reported ±CI is over seeds (5 sessions). Within a combination, per-seat means differ by an order of magnitude more than the cross-seed CI (six_loose: seat means span −408 to −4.6; the CI is ±150). Cross-seat variance, not seed variance, is the dominant uncertainty, and no margin of that kind is shown in the headline table.
4. **Monte Carlo equity at 400 iterations.** Postflop equity is an estimate. Thin pot-odds decisions — Mathematician's bread and butter — are the most sensitive, and this interacts with (1): the style most dependent on exact equity is the style whose tables die fastest.
5. **No-bluff bound.** The six styles cannot deceive, and the pot-limit cap already forbids the over-bet-as-bluff lever. Everything here is a statement about *fixed, honest distributions of play*; a real no-limit game with bluffing and game-theoretic mixing is out of scope.
6. **Conservation trap.** No rake and no rebuys mean a mean of zero is the only possible equilibrium; the report's ordering is over *loss minimization and survival*, not over profit. The natural next experiment is to add an economy that gives profit meaning.
7. **Reproducibility is claimed only within these rules.** Byte-identical logs require identical `seed + config`; changing any rule (blinds, cap, MC iterations, stack size) changes the digest and the numbers. The committed digest is `21f78e32923c9779`.

## 7. Conclusion and recommended next experiments

**The honest answer to the core question:** in this no-rake, no-rebuy, dying-table, pot-limit regime, the least-staking, tightest style (six_tight, BB/100 −2.5) both *loses the least* and *outlives every other table*; every looser or EV-heavy style, including the mathematically correct one, busts its own table's statistical power so completely that "most profitable" is confounded with "outlives the horizon". Mixing styles does not rescue profitability — `benchmark_mix` and the 3-and-3 splits land mid-table, and the aggressive-inclusive mixes inherit the ruin of their volatile seats.

The simulator is deterministic and auditable, so these results are *reproducible, not asserted*; but before trusting the ordering as a statement about real poker, the design must stop punishing variance. Ordered by expected leverage:

1. **Rebuy / implicit-rake model** — gives profit a meaning and stops tables from self-destructing; the single most important change.
2. **Fixed-horizon sessions** (stop at N hands regardless of survivors) — decouples style winrate from table longevity and fixes the effective-hand-count falloff.
3. **No-limit raises** — lets Aggressive/Adaptive test the levers the pot-cap currently bans.
4. **Bluff and mixed strategies** — without these, the "no bluff" bound is airtight and a true long-run answer is impossible.
5. **More seeds + more MC iterations** — narrows the cross-seed CI and the equity error; important but pointless until (1) makes tables live long enough to benefit.
6. **Persist and roll up `positional_ev` and starting-hand aggregates to CSV** — turns sections 4 and 5 from interpretations into measurements.