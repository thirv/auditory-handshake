# Auditory handshake — developer harness (S2.1).
#
# Toolchain (installed via `pip install -e ".[dev]"`, see pyproject.toml):
#   ruff   — linter + import sorting      (config: [tool.ruff])
#   black  — formatter                    (config: [tool.black], line-length 100)
#   pytest — tests                        (config: [tool.pytest.ini_options])
#   mypy   — strict type checker          (config: [tool.mypy])
#
# Design goal: a *sub-second* feedback loop. `lint` and `test` run only the fast
# tools; `mypy` is a separate `types` target (slower cold start) and is part of
# the full `check` gate. All tools are invoked through `python -m ...` so the
# harness works regardless of whether the CLI shims are on PATH (conda env / venv).

.DEFAULT_GOAL := check

PY     := python
RUFF   := $(PY) -m ruff
BLACK  := $(PY) -m black
PYTEST := $(PY) -m pytest
MYPY   := $(PY) -m mypy

# Python source dirs. ruff / black / mypy / pytest all pick up their config from
# pyproject.toml (run from the repo root), so no flags are repeated here.
PACKAGES := src tools tests

.PHONY: help check lint test fmt types cfg clean sim-01 sim-02 sim-02b sim-03 sim-04 sim-05 sim-07 sim-08 sim-09 sim-10

help: ## List available targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_.-]+:.*##/ {printf "  \033[36m%-8s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

check: lint test ## Full fast gate (the S2 acceptance bar: lint + test)

lint: ## Lint (ruff) and verify formatting (black --check)
	$(RUFF) check $(PACKAGES)
	$(BLACK) --check --diff $(PACKAGES)

test: ## Run the test suite
	$(PYTEST)

fmt: ## Format (black) and auto-fix lint issues (ruff --fix)
	$(BLACK) $(PACKAGES)
	$(RUFF) check --fix $(PACKAGES)

types: ## Type-check with mypy (strict)
	$(MYPY)

cfg: ## Sanity-check configs: load defaults + every room preset (S2.4)
	$(PY) -m auditory_handshake.config

cap: ## Print the Shannon capacity table from PLAN.md §2.1 (S3.1)
	$(PY) -m auditory_handshake.metrics.capacity

sim-01: ## Run the S6 link-budget sim: capacity vs distance (PLAN §2.5)
	$(PY) sims/01_link_budget.py

sim-02: ## Run the M2 channel-characterisation sim: RT60 / τ_rms vs frequency (PLAN §12.5)
	$(PY) sims/02_channel_characterization.py

sim-02b: ## Run the S12.2 motion & Doppler sim: shift vs velocity (PLAN §12.2)
	$(PY) sims/02b_motion_doppler.py

sim-03: ## Run the S13 CSS vs MFSK A/B sim: BER vs SNR per room (PLAN §4, §12)
	$(PY) sims/03_css_vs_mfsk_ab.py

sim-04: ## Run the S14 preamble-detection latency/FAR sweep (PLAN §4, §12)
	$(PY) sims/04_preamble_detection.py

sim-05: ## Run the S14 preamble-detection CSS integration sim (PLAN §4, §12)
	$(PY) sims/05_preamble_integration.py

sim-07: ## Run the M6 TTCH N-device slotted-ALOHA contention model (PLAN §2.4, C1)
	$(PY) sims/07_n_device_contention.py

sim-08: ## Run the S15 2-tap RAKE A/B: MFSK BER vs SNR, 2 extreme-RT60 rooms (PLAN §12)
	$(PY) sims/08_rake_2tap.py

sim-09: ## Run the C1 two-way handshake validation: measured RTT vs measured RT60 (PLAN §3, §12)
	$(PY) sims/09_two_way_handshake.py

sim-10: ## Run the C2 frontier sim: goodput vs audibility cost under (BLER, TTCH) budgets (PLAN §2.4, §3)
	$(PY) sims/10_c2_frontier.py

clean: ## Remove caches and bytecode (does not touch the editable install)
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@echo "cleaned."