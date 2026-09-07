# Technical Document: Texas Hold'em Simulator for Strategy Analysis (No Bluff)

## 1. Goals and Objectives

Develop a cash-game simulator for Texas Hold'em with 6 players, each controlled by one of six strategies:

1. **Tight**  
2. **Loose**  
3. **Aggressive**  
4. **Passive**  
5. **Mathematician**  
6. **Adaptive**

Investigate the influence of playing style on expected value (EV), probability of ruin, and stack dynamics. Run tournaments between various combinations of strategies (e.g., 6 loose, 3 tight + 3 aggressive, etc.) and analyze the collected metrics.

**Model limitations:**  
- No bluffing (bets are made based on actual hand strength, expressed through equity).  
- Players do not know each other's algorithms; decisions are made only on the basis of their own cards, community cards, and the observable history of the current hand (bets, folds).  
- Equity calculation is against random opponent hands (uniform distribution among all possible unknown cards).  
- Cards folded remain unknown until the end of the hand.  
- Only known dead cards (own cards and community cards) are taken into account.

---

## 2. Game Rules

### 2.1. General Hand Structure
- **Deck:** standard 52 cards.
- **Players:** 6.
- **Blinds:** small blind (SB) = 1 chip, big blind (BB) = 2 chips.
- **Initial stack per player:** 200 chips (100 BB).
- **Betting rounds:** 4 rounds – preflop, flop, turn, river.
- **Maximum raises per round:** 4 (including the initial bet, i.e., bet + 3 raises).

### 2.2. Pot-Limit
- Minimum bet = size of the big blind (2 chips).
- Minimum raise = size of the previous bet or raise (on the preflop – the big blind).
- Maximum raise = current pot size + 2 × (amount to call).  
  Formula: `MaxRaise = pot + 2 * call_amount`.  
  After calling, the player can raise by the size of the resulting pot (including his call), but the total bet cannot exceed this amount.

### 2.3. Order of Action
- After the cards are dealt: preflop – starting with the player to the left of the BB (UTG), then clockwise.
- On the flop, turn, river: first to act is the player to the left of the button (SB).
- If all but one player fold, the remaining player wins the pot without showdown.
- If after the river two or more players remain – showdown: cards are revealed, the pot is split (in case of equal hands – equally).

### 2.4. Stack Persistence
- Cash game: after each hand, the player's stack changes by the amount won/lost.
- If a player's stack becomes less than the size of the big blind, he is eliminated from the table (ruin). In simulation, such a player is removed, his position is skipped, and to maintain a full table of 6 players a new player with a starting stack can be added (or the session can be terminated – a configurable parameter; in the basic version, the session ends when any player is eliminated, unless otherwise specified).

---

## 3. Data Model and Core Structures

### 3.1. Cards and Deck
- Card: tuple `(rank, suit)`, rank 2–14 (Ace = 14), suit 0–3.
- Deck: list of 52 cards, shuffled before each hand (fair Fisher–Yates).

### 3.2. Hand Combinations
- Evaluation of a 5-card hand: standard function `evaluate_hand(cards)` returning a tuple `(category, kickers)`, where category is from 0 (high card) to 8 (straight flush).  
- For 7 cards (2 hole + 5 community), the best 5 of 7 is selected (enumeration of 21 combinations).

### 3.3. Hand State
- `players`: list of 6 player objects with stacks, cards, statuses (active/folded/all-in).
- `board`: community cards (0, 3, 4, 5).
- `pot`: current pot.
- `current_bet`: current bet in the current betting round.
- `bets`: dictionary of each player's bets in the current round.
- `dealer_pos`: position of the button (0–5), shifts clockwise after each hand.
- `hand_history`: list of actions (for adaptive strategy and analysis).

### 3.4. Hand Logging
For each hand, the following are saved:
- Positions of players, their stacks before/after.
- Hole cards of each player (for analysis, but not for bots).
- Community cards, actions on each round (player, action, size).
- Final win/loss.

---

## 4. Equity Calculation

**Definition:** probability of winning the hand (including ties) against a given number of opponents with random cards from the remaining unknown cards.

### 4.1. Preflop
On the preflop, equity depends only on the starting hand (2 cards) and the number of opponents. **Precomputed tables** for all 169 types of starting hands against 1, 2, ..., 5 random opponents are used.  
- The table is generated once by full enumeration or loaded from a file.  
- Dead cards (your two cards already removed from the deck) are taken into account.

### 4.2. Postflop (flop, turn, river)
Exact enumeration of all possible opponent hands and remaining board cards can be expensive. For acceleration, a **hybrid method** is used:
- If opponents ≤ 2 and known community cards ≥ 3, exact enumeration is possible (optimized with caching) – but in the basic version we apply the **Monte Carlo method** with a fixed number of simulations (e.g., 300–500 per evaluation) to maintain speed with 5 opponents.
- **Monte Carlo equity algorithm:**
  1. Remove known cards (own and community) from the deck.
  2. For each simulation:
     - Randomly deal 2 cards to each opponent.
     - Deal remaining community cards.
     - Determine the best hand for us and for each opponent.
     - If our hand is stronger than all – count as win; if equal to someone – tie (share 0.5).
  3. Equity = (wins + 0.5 * ties) / number of simulations.
- Accuracy is sufficient for decision-making; if necessary, the number of simulations can be increased.

**Optimization:**  
- Use `numpy` for vectorized generation of random hands.  
- Cache results for identical states (own hand + board + number of opponents) within a single hand if the state repeats.

---

## 5. Player Strategies

Each strategy is implemented as a function of the current state and observable information. Inputs:  
- `hand`: own cards,  
- `board`: community cards,  
- `equity`: current equity against the number of active opponents (calculated dynamically),  
- `pot_odds`: pot odds for a call (required winning share for break-even),  
- `position`: relative position (early/middle/late/blinds),  
- `action_history`: history of the current hand (for adaptive).  

Formal rules for each strategy are given below. Thresholds are chosen empirically and may be refined.

#### 5.1. Tight
- **Preflop:**  
  Plays only strong starting hands: pairs 77+, AQo+, AJs+, KQs. (Top ~12–15% of hands).  
  In early position – only premium (JJ+, AK).  
  In late position, the range is slightly wider.
- **Postflop:**  
  Continues only with equity ≥ 0.6 on the flop, ≥ 0.7 on turn/river.  
  Bets/raises only with very strong hands (equity ≥ 0.75) or with a made hand from two pairs and above.  
  Folds to aggression if equity below the call threshold according to pot odds.

#### 5.2. Loose
- **Preflop:**  
  Plays ~40% of starting hands: any pairs, any suited connectors, A2s+, KTo+, QJo+, suited kings and queens.  
  Often calls, rarely raises himself (only with top 10%).
- **Postflop:**  
  Calls with equity ≥ 0.35 (or if has a draw).  
  Bets/raises only with made hands (pair+).

#### 5.3. Aggressive
- **Preflop:**  
  Plays ~30% of hands, but often raises (with hands from top 20%).  
  In position, steals (raises with hands like 45s, suited connectors) to push opponents out.  
- **Postflop:**  
  Makes a continuation bet (c-bet) almost always (if was the preflop aggressor), regardless of hitting the board.  
  Raises with any hand having equity ≥ 0.5, or with a good draw.  
  Rarely calls, prefers aggression.

#### 5.4. Passive
- **Preflop:**  
  Plays ~25% of hands, but almost never raises (only with AA, KK, QQ, AK).  
  Calls raises with hands from top 15–20%.
- **Postflop:**  
  Rarely bets himself, mainly calls.  
  Calls bets if equity ≥ pot odds (or slightly lower, due to passivity).  
  Not prone to bluffing, raises only with very strong hands (set+).

#### 5.5. Mathematician
- **All streets:**  
  Decision strictly based on comparison of current equity with pot odds.  
  If `equity ≥ pot_odds` – call, otherwise fold.  
  Raises only if equity is significantly higher (e.g., 15%+ above pot odds) and the raise size is mathematically justified (increases EV).  
- In practice, this means he will call many draws with correct odds and fold to overly large bets.
- **Betting threshold:** if equity > 0.6, he bets 50–75% of the pot.

#### 5.6. Adaptive
- **Tracks opponent statistics during the current session:**
  - Fold to Raise % for each player.
  - C-bet % on the flop.
  - Overall aggression (ratio of bets/raises to calls).
- **Preflop:**  
  Base range – top 20%.  
  If many tight players at the table (often fold to raise), expands raising range to 30% and increases steal size.  
  If many loose/calling players, tightens to 15% and bluffs less.
- **Postflop:**  
  Uses information about opponents' fold tendencies: if an opponent often folds to a bet, c-bets more frequently, even with air.  
  If an opponent is aggressive, plays tighter, calls only with strong hands.
- **Thresholds are dynamically adjusted every N hands (e.g., every 20 hands)** based on accumulated statistics.

**Table 1. Basic preflop thresholds (approximate ranges)**

| Strategy   | Playable range (approx.) | Raise frequency (top % hands) | Note |
|------------|--------------------------|-------------------------------|------|
| Tight      | 12–15%                   | 5% (JJ+, AK)                  | Early position even tighter |
| Loose      | 40%                      | 10%                           | Many calls |
| Aggressive | 30%                      | 20%                           | Often steals |
| Passive    | 25%                      | 2% (AA,KK,QQ,AK)              | Almost never raises |
| Mathematician | 20–25% (by pot odds)  | by EV                         | Strictly mathematical |
| Adaptive   | 20% (base, changes)      | dynamic                       | Adjusts to table |

**Table 2. Postflop equity thresholds for calling (against 1 opponent)**

| Strategy   | Flop  | Turn  | River |
|------------|-------|-------|-------|
| Tight      | ≥ 0.55| ≥ 0.65| ≥ 0.75|
| Loose      | ≥ 0.35| ≥ 0.40| ≥ 0.50|
| Aggressive | ≥ 0.45| ≥ 0.50| ≥ 0.55|
| Passive    | ≥ 0.40| ≥ 0.45| ≥ 0.55|
| Mathematician | by pot odds | by pot odds | by pot odds |
| Adaptive   | dynamic (usually ≥ 0.4)| dynamic | dynamic |

*Note: with an increase in the number of opponents, thresholds rise (e.g., with 3 opponents, required equity for a call increases approximately 1.5–2 times).*

---

## 6. Simulation Process

### 6.1. Session Setup
- A combination of strategies for 6 seats is chosen (e.g., `[Tight, Loose, Aggressive, Passive, Mathematician, Adaptive]`).  
- Players receive initial stacks of 200 chips.  
- The initial button position is randomly chosen.  
- Number of hands per session – a parameter (recommended from 10,000 to 50,000 for statistical significance).

### 6.2. Hand Cycle
1. Shuffle the deck.  
2. Deal 2 cards to each active player.  
3. Conduct preflop betting round.  
4. If ≥ 2 players remain and no all-in, deal 3 flop cards.  
5. Betting round.  
6. Turn (1 card), betting round.  
7. River (1 card), betting round.  
8. Showdown (if needed) and pot distribution.  
9. Update stacks. If any stack ≤ 0 – player eliminated (session may stop or continue with a new player).  
10. Move the button.

### 6.3. Betting Round Processing
- Active player determined by order.
- Available actions: fold, check (if no bet), call (if there is a bet), bet/raise (if raise limit allows).
- Player's algorithm `act(state)` returns the chosen action.
- After each action, pot, current bets, history are updated.
- Round ends when all active players have matched the bets or folded (or the maximum raises have been reached and all called).

### 6.4. Parallelism and Acceleration
- Each session (strategy combination) can be run in a separate process.
- Within a session, hands cannot be trivially parallelized due to shared stack state, but independent copies of the session with different seeds can be simulated and averaged.
- Use `multiprocessing.Pool` to run multiple sessions concurrently.
- Hot functions (equity evaluation, evaluate_hand) optimized with `numba` (JIT) or Cython.

---

## 7. Data Collection and Analysis

### 7.1. Metrics for Each Session
- **Average win/loss** in big blinds per 100 hands (BB/100) for each player.
- **Winrate** – percentage of hands won.
- **Ruin probability** – probability that a player with a given strategy loses the entire stack before the end of the session (for a fixed number of hands).
- **Final stack distribution** (histogram, mean, median).
- **Profitability of starting hands** (13×13 heatmap) for each strategy: average win in BB for each pair.
- **Positional metrics:** EV for each position (SB, BB, UTG, MP, CO, BTN).
- **Aggression:** frequency of bets, raises, calls, folds for each strategy.

### 7.2. Output Format
- All metrics saved to CSV/JSON for further analysis.
- Graphs built with `matplotlib`/`seaborn` (heatmaps, stack boxplots, stack evolution lines).
- For comparison of strategy combinations, summary tables and charts are generated.

### 7.3. Experiment Plan
Run sessions for various combinations, e.g.:
- All 6 of one strategy (6 tight, 6 loose, etc.).
- Mixed: 3 tight + 3 loose, 2 aggressive + 4 passive, one of each of the six.
- For each combination, conduct 10,000–50,000 hands (multiple repetitions with different seeds to estimate variance).

---

## 8. Technical Implementation

### 8.1. Architecture (Modules)
- `card.py` — Card, Deck classes.
- `hand_evaluator.py` — hand evaluation function.
- `player.py` — base Player class and subclasses for each strategy.
- `equity.py` — equity calculation (preflop tables, Monte Carlo).
- `game.py` — hand and betting engine.
- `simulator.py` — session management, hand loop, statistics collection.
- `analysis.py` — post-processing, metric computation, visualization.
- `config.py` — settings (blinds, stacks, raise limit, number of Monte Carlo simulations, etc.).

### 8.2. Key Functions
- `evaluate_hand(cards: List[Card]) -> Tuple[int, ...]` — returns hand rank.
- `calc_equity(hand, board, num_opponents, mc_iterations=400) -> float`.
- `Player.act(state: GameState) -> Action` — makes a decision.
- `run_hand(players, dealer_pos, deck) -> HandResult`.
- `run_session(strategy_list, num_hands, seed) -> SessionStats`.

### 8.3. Performance Optimization
- Preflop equity tables loaded once into memory (size ~169×5).
- `evaluate_hand` vectorized for arrays of cards (for Monte Carlo).
- Use `numba.jit` for `evaluate_hand` and random card generation.
- Parallel session execution via `multiprocessing`.

### 8.4. Dependencies
- Python 3.9+
- `numpy` — vector operations.
- `numba` — JIT compilation of hot functions.
- `pandas` — data aggregation.
- `matplotlib`, `seaborn` — plotting.
- `tqdm` — progress bars.
- `multiprocessing` (standard library).

---

## 9. Implementation Stages

1. **Stage 1: Game Core**  
   - Implement cards, deck, hand evaluation.  
   - Betting engine with pot-limit and 4 raises.  
   - Testing on manual scenarios.

2. **Stage 2: Equity Calculation**  
   - Generate/load preflop tables.  
   - Implement Monte Carlo for postflop.  
   - Optimize with numba.

3. **Stage 3: Basic Strategies**  
   - Implement first four strategies (tight, loose, aggressive, passive).  
   - Run trial sessions, verify correctness.

4. **Stage 4: Additional Strategies**  
   - Implement "Mathematician" and "Adaptive".  
   - Test in single and mixed sessions.

5. **Stage 5: Data Collection and Analysis**  
   - Log all actions and results.  
   - Compute metrics, build heatmaps and graphs.  
   - Run mass simulations for all combinations.

6. **Stage 6: Optimization and Finalization**  
   - Profiling, speeding up bottlenecks.  
   - Final reports.

---

## 10. Risks and Assumptions

- **Monte Carlo accuracy:** with 400 iterations, equity error can be ±0.05, which affects decisions. To improve accuracy, iterations can be increased or analytical methods used for common situations.  
- **Computational load:** 50,000 hands with 6 players and equity evaluation on each action can be slow. Optimization and parallelism are critical.  
- **Strategies without bluffing may be unrealistic:** absence of bluffing simplifies the model but isolates the effect of hand strength. Bluffing can be added later if desired.  
- **Pot-limit and 4 raises:** rules must be strictly followed to avoid pot errors.  
- **Adaptive strategy** may require storing a large amount of history, which could slow simulation; the depth of tracked statistics should be limited (e.g., last 100 hands).

---

## 11. Conclusion

This simulator will allow quantitative comparison of different playing styles in Texas Hold'em under the condition of no bluffing. The collected metrics (EV, winrate, hand heatmaps, ruin probability) will provide insight into which strategy is most profitable in the long run and how mixing styles affects results. The technical implementation using Python, numpy, numba, and multiprocessing will ensure sufficient performance to run tens of thousands of hands and analyze the data.