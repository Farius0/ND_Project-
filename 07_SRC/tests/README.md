# Tests

This directory contains **lightweight test suites** to validate operators, datasets, and user‑facing behaviors.  
Tests are organized in three layers to separate quick unit checks, development validations, and user interactions.

## Structure

- `experiments/`  
  Exploratory or prototype tests (not strictly unit tests). Useful for trying new ideas on small samples.

- `test_dev/`  
  Developer‑oriented checks for **operators and utilities** (fast, repeatable).  
  Aim: catch regressions early while coding (CI‑friendly).

- `tests_users_interact/`  
  Integration tests mimicking **user workflows** (I/O, transforms sync, datasets, simple training loops).  
  Slower but closer to real usage.

## Conventions

- **Frameworks**: tests must support both **NumPy** and **PyTorch** when relevant.  
- **Layouts**: always specify the layout explicitly (`HW`, `HWC`, `NCHW`, `DHW`) to avoid ambiguity.  
- **Seeds**: use fixed seeds for reproducibility (`np.random.seed`, `torch.manual_seed`).  

## Running

We recommend `pytest`:
```bash
pytest -q
# or focused:
pytest test_dev/test_diff_operator.py -q
```
## Tips

- Prefer **small shapes** (e.g., 32×32) and **few channels** to keep tests fast.  
- If a test requires optional dependencies, **skip** it with `pytest.importorskip("pkg")`.  
- Keep user‑interaction tests **deterministic** (fixed seeds, fixed splits).

