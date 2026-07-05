# 2D FDTD Electromagnetic Wave Simulation Framework

A pedagogical Python-based 2D FDTD (finite-difference time-domain) simulation framework for physics education, covering five canonical electromagnetic wave phenomena.

## Features

- **Five simulation scenarios**: free-space propagation, PEC reflection, dielectric slab transmission, double-slit interference, parallel-plate waveguide
- **Minimal dependencies**: pure Python with only NumPy and Matplotlib
- **Educational transparency**: ~200 lines of core solver, each line mapping to a Maxwell equation
- **Cross-platform**: works on Windows, macOS, Linux, and Google Colab

## Requirements

- Python ≥ 3.10
- numpy ≥ 1.24
- matplotlib ≥ 3.7

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all five simulation scenarios
python3 fdtd_2d_tm.py

# Generate publication-quality figures
python3 generate_figures.py

# Run convergence analysis
python3 run_convergence.py
```

## File Structure

```
code/
├── fdtd_2d_tm.py            # Core FDTD solver + all scenario runners
├── common_utils.py          # Shared utility functions  
├── generate_figures.py      # Figure generation pipeline
├── run_convergence.py       # Convergence analysis
├── compute_convergence_ci.py# CI calculation for convergence exponent
├── measure_quantities.py    # Physical quantity measurements
├── measure_transmission_*.py# Transmission coefficient measurements
├── requirements.txt         # Python dependencies
├── README.md                # This file
```

## Simulation Scenarios

| # | Scenario | Source Type | Grid Size | Key Physics |
|---|----------|-------------|-----------|-------------|
| 1 | Free-space propagation | Gaussian pulse | 150×150 | Wavefront evolution, Mur ABC |
| 2 | PEC reflection | Gaussian pulse | 150×150 | Standing waves, π phase shift |
| 3 | Dielectric slab transmission | Modulated Gaussian | 200×150 | Fresnel equations, Fabry-Pérot |
| 4 | Double-slit interference | Sinusoidal (2 GHz) | 200×300 | Fraunhofer diffraction |
| 5 | Parallel-plate waveguide | Modulated Gaussian | 300×80 | TM₁ mode, cutoff frequency |

## Customization

To add a new scenario, create a function in `fdtd_2d_tm.py`:

```python
def run_my_scenario():
    f = FDTD2DTM(nx=200, ny=200, dx=5e-3)
    f.set_source(40, 100)
    # Configure materials, boundaries...
    result = f.run(500, 'modulated', save_interval=10)
    return result
```

## License

MIT License — free to use, modify, and distribute.
