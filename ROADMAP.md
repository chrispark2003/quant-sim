# quant-sim Roadmap

**SIMULATED / PAPER TRADING ONLY -- NOT FINANCIAL ADVICE.**

This roadmap tracks the shift from a quant-grade paper trading simulator into
a training platform where trading agents can learn, compete, and be evaluated
inside realistic simulated markets. The project must remain simulation-only:
no component should connect to real brokerage or exchange order-placement
endpoints.

## V1: Agent Training Environment

V1 turns the current backtesting and simulated execution stack into a native
Python environment for trainable quant and crypto agents.

Planned capabilities:

- Add an `agents/` package with a small agent protocol:
  - Agents receive an observation object.
  - Agents return target portfolio actions.
  - Include baseline agents for smoke tests: hold, random, and an adapter for
    the existing signal ensemble.
- Add an `environment/` package with `TradingEnvironment`:
  - Owns episode `reset` and `step` flow.
  - Replays historical bars through the existing execution simulator and
    virtual ledger.
  - Exposes observations with recent features, prices, positions, cash,
    equity, realized volatility, and timestamp.
  - Accepts actions as target portfolio weights or target notionals per symbol.
  - Computes rewards from equity return with penalties for drawdown, turnover,
    fees, and leverage.
- Extend bar-level execution realism:
  - Preserve current slippage, fees, latency, and volume participation caps.
  - Add configurable bid/ask spread modeling.
  - Add explicit partial fill behavior.
  - Add funding or borrow costs where relevant.
  - Add exchange/session availability rules.
- Add local experiment orchestration:
  - Run experiments across agents, symbols, date ranges, and seeds.
  - Persist run config, agent name, seed, metrics, equity curve, blotter, and
    summary ranking.
  - Add a CLI entrypoint such as `scripts/run_experiment.py`.
- Add experiment configuration:
  - Add `config/experiments.yaml`, or extend existing config if that keeps the
    structure simpler.
  - Keep current backtest, live loop, dashboard, signal, risk, and execution
    behavior intact unless explicitly used by experiments.

V1 success criteria:

- A baseline agent can run a deterministic historical episode end to end.
- Multiple local experiment runs can be compared by metrics and equity curve.
- Existing backtest, execution, risk, signal, and dashboard tests continue to
  pass.
- The implementation remains paper/simulated only.

## V2: RL Library Compatibility

Expose the environment through a thin Gymnasium-compatible wrapper while
keeping the native domain model as the source of truth.

Planned capabilities:

- Add optional Gymnasium dependency support.
- Provide `reset`, `step`, `render`, observation space, and action space.
- Document integration examples for common RL training loops.
- Keep the wrapper thin so future environment changes do not require rewriting
  the core trading model.

## V3: Order-Book Simulation

Add a richer market simulator for agents that need bid/ask depth and liquidity
dynamics beyond OHLCV bars.

Planned capabilities:

- Support L2-style order book snapshots or synthetic book generation.
- Model spread, depth, market impact, queue fill probability, and limit-order
  placement.
- Allow the environment to choose between bar-level execution and order-book
  execution by configuration.

## V4: Distributed Orchestration

Scale experiment orchestration beyond one local process once the agent and
environment APIs are stable.

Planned capabilities:

- Add worker processes or distributed workers.
- Add job queueing, retry behavior, run state tracking, and artifact syncing.
- Support larger sweeps across agents, reward settings, symbols, and seeds.

## V5: Live Paper-Agent Supervision

Use trained agents in the existing live paper-trading loop under strict
simulation guardrails.

Planned capabilities:

- Load selected trained agent artifacts into paper-trading mode.
- Add supervision policies, kill-switch integration, and dashboard visibility.
- Track live paper-agent decisions separately from historical training runs.
- Preserve the rule that all fills remain simulated against the virtual ledger.

## Later Research Tracks

These are deliberately out of V1 scope but worth preserving for future design:

- Exchange microstructure simulation with matching-engine behavior, queue
  priority, maker/taker dynamics, and event-stream replay.
- Multi-agent market simulation where agents can interact in the same synthetic
  venue.
- Synthetic stress regimes for liquidity shocks, volatility spikes, outages,
  stale data, liquidation cascades, and correlated market breaks.
- Model evaluation tooling for robustness, overfitting detection, regime
  generalization, and policy explainability.

## Not In V1

V1 does not include:

- Real exchange or brokerage order placement.
- Distributed workers.
- Full exchange microstructure simulation.
- L2/L3 order-book replay.
- Live deployment of trained agents.
- Mandatory Gymnasium support.

