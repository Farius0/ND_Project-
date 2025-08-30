# ND Project

ND Project is a research-oriented Python package for image processing, analysis, and segmentation.  
It is designed to be modular, extendable, and compatible with PyTorch and NumPy, while also supporting custom operators and pipelines.

---

## Project Structure

```
(see detailed tree structure in folder "07_SRC")

```

---

## Installation

### 1. Clone the repository
```bash
git clone https://github.com/Farius0/ND_Project-.git
cd nd_project
```

### 2. Create and activate a virtual environment
```bash
python -m venv .venv
# On Linux/MacOS
source .venv/bin/activate
# On Windows (PowerShell)
.venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -U pip
pip install -r requirements/requirements.txt
```

### 4. Install project in editable mode
```bash
pip install -e .
```

---

## Quick Test

After installation, check that everything works:

```bash
python -c "from operators.feature_extractor import FeatureExtractorND; print('OK:', FeatureExtractorND.__name__)"
```

Expected output:
```
OK: FeatureExtractorND
```

---

## Running Tests

```bash
pytest -q
```

---

## Requirements Files

`07_SRC/requirements/`  
- minimal dependencies (`numpy`, `scipy`, `scikit-image`, etc.)
- PyTorch CPU version or GPU (Optional)

---

## Features

- Modular operator system (differential operators, edge detectors, filters…)
- Compatible with both NumPy and PyTorch backends
- Advanced image analysis: texture descriptors (GLCM, entropy), denoising, segmentation
- Dataset generators for deep learning pipelines
- Logging and testing utilities

---

## Development Notes

- Each subfolder has an `__init__.py` → all are proper Python packages/modules
- The project follows a modular architecture: every operator/filter/algorithm can be reused independently
- Config files are centralized in `core/config.py`

---

## License

This project uses a **dual-license model**:

- **MIT License** for generic and reusable components (`core/`, `utils/`, `operators/`, `filters/`, `algorithms/`, `datasets/`).  
- **Proprietary License** for research-specific parts (`scripts/`, `results/`, `logs/`).  

See the [`LICENSE`](LICENSE) file for details.

---

## Author

Developed by **Farius Aina**  
This project is part of the *Stage LC-OCT* research on skin imaging and segmentation.
