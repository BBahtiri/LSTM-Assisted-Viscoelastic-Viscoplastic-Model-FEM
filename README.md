# Hybrid ML-FEM Viscoelastic-Viscoplastic Damage Model

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.cma.2023.116293-blue)](https://doi.org/10.1016/j.cma.2023.116293)
[![License](https://img.shields.io/badge/License-GPL--2.1-green.svg)](LICENSE)
[![deal.II](https://img.shields.io/badge/deal.II-9.0%2B-orange.svg)](https://dealii.org/)

A sophisticated finite element implementation combining traditional constitutive modeling with machine learning (LSTM neural networks) for modeling complex material behavior in epoxy nanocomposites under cyclic loading conditions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Model Description](#model-description)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Output](#output)
- [Examples](#examples)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

This code implements a **hybrid ML-FEM approach** for large deformation solid mechanics, specifically designed for epoxy nanocomposite materials with moisture content. The implementation seamlessly combines:

- **Traditional physics-based constitutive models** for initial simulation phases
- **LSTM neural networks** for accelerated computation in later phases
- **Multi-network viscoelastic-viscoplastic formulation** with damage evolution
- **Nanoparticle and moisture effects** through amplification factors

The approach is based on deal.II's `step-44` tutorial and extends the `Quasi_static_Finite_strain_Compressible_Elasticity` example from the deal.II code gallery.

![Rheological Model](/rheo.PNG)

## Features

### 🔧 **Multi-Physics Material Model**
- **Viscoelastic behavior**: Two-network model with equilibrium and non-equilibrium responses
- **Viscoplastic behavior**: Rate-dependent plasticity with backstress evolution
- **Damage mechanics**: Progressive material degradation under cyclic loading
- **Environmental effects**: Temperature and moisture content dependencies
- **Nanoparticle reinforcement**: Weight fraction-dependent amplification

### 🤖 **Machine Learning Integration**
- **LSTM neural networks**: Two-layer architecture with 200 hidden units
- **Seamless switching**: Automatic transition from physics to ML models
- **Pre-trained weights**: Load neural network parameters from external files
- **Numerical tangent**: Automatic differentiation for ML components

### ⚡ **Advanced Numerical Methods**
- **Updated Lagrangian formulation**: Large deformation framework
- **Newton-Raphson solver**: Robust nonlinear solution scheme
- **Parallel processing**: MPI support for large-scale computations
- **Adaptive timestepping**: Dynamic time step control

### 📊 **Comprehensive Analysis**
- **Cyclic loading**: Progressive amplitude cycling with unloading
- **Multi-scale output**: Element and nodal averaging options
- **Real-time monitoring**: Force-displacement tracking at specific points
- **Paraview visualization**: VTK output for post-processing

## Model Description

### Rheological Framework

The material model is based on a multi-network rheological representation:

```
Total Response = Equilibrium Network + Viscous Network + Viscoplastic Element + Damage
```

- **Equilibrium Network (F_e, σ_eq)**: Instantaneous elastic response
- **Viscous Network (F_v, σ_neq)**: Time-dependent Maxwell element behavior  
- **Viscoplastic Element (F_vp)**: Rate-dependent plastic flow with threshold
- **Damage Factor (1-d)**: Progressive stiffness reduction

### Mathematical Formulation

**Multiplicative decomposition:**
```
F_total = F_ve × F_vp^(-1)
F_ve = F_e × F_v^(-1)
```

**Viscous flow evolution:**
```cpp
γ̇ = γ̇_0 × exp[(ΔG/kT) × (τ/τ_hat)^m - 1]
```

**Viscoplastic flow:**
```cpp
γ̇_p = a × |ε - ε_0|^b × ε̇  (when τ > σ_0)
```

**Environmental amplification:**
```cpp
X = (1 + 5φ_np + 18φ_np²) × α_Z × α_T
```

### Hybrid ML-Physics Approach

```cpp
if (timestep < switch_point) {
    // Traditional constitutive model
    update_internal_equilibrium();
} else {
    // LSTM neural network
    lstm_forward();
}
```

## Installation

### Prerequisites

- **deal.II 9.0+** with MPI support
- **CMake 3.10+**
- **C++ compiler** with C++14 support
- **Eigen3** linear algebra library
- **Optional**: Trilinos with Sacado for automatic differentiation

### Build Instructions

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/vevp-ml-model.git
cd vevp-ml-model
```

2. **Configure and compile:**
```bash
cmake -DDEAL_II_DIR=/path/to/deal.II .
make release
make
```

3. **Create output directory:**
```bash
mkdir output
```

### Verify Installation

```bash
./vevp_ml_model --help
```

## Usage

### Basic Execution

```bash
./vevp_ml_model
```

The program reads parameters from `parameters.prm` and generates:
- **VTK files**: In `output/` directory for Paraview visualization
- **Force-displacement data**: `data-for-gnuplot.sol` for plotting
- **Timing information**: `data_times.txt` for performance analysis

### MPI Execution

```bash
mpirun -np 4 ./vevp_ml_model
```

### Physics-Only Mode

To run without machine learning (recommended for initial testing):

```bash
# In parameters.prm:
set Switch to ML = Off
set At which timestep switch to ML ? = 2147483646
```

### ML-Enhanced Mode

To enable machine learning acceleration:

```bash
# In parameters.prm:
set Switch to ML = On
set At which timestep switch to ML ? = 500
```

**Note**: ML mode requires pre-trained LSTM weight files in the working directory.

## Configuration

### Key Parameter Categories

#### **Loading Conditions**
```
subsection Boundary conditions
  set Load type = cyclic_to_zero        # Cyclic loading with unloading
  set First stretch = 0.5               # Progressive amplitudes (mm)
  set Second stretch = 0.7
  # ... up to Seventh stretch
  set Total cycles = 7
end
```

#### **Material Properties**
```
subsection Material properties
  # Hyperelastic networks
  set mu1 = 760.0                       # Equilibrium network modulus (Pa)
  set mu2 = 790.0                       # Viscous network modulus (Pa)
  set nu1 = 0.23                        # Poisson's ratio
  
  # Viscous flow
  set gamma_dot_0 = 9.7746e11           # Reference strain rate
  set dG = 1.9761e-19                   # Activation energy
  set m = 0.657                         # Rate sensitivity
  
  # Viscoplasticity
  set sigma0 = 5.5                      # Yield threshold (Pa)
  set a = 0.179                         # Flow parameter
  set b = 0.910                         # Flow exponent
  
  # Environmental effects
  set temp = 296.0                      # Temperature (K)
  set zita = 0.0                        # Water content
  set wnp = 0.0                         # Nanoparticle weight fraction
end
```

#### **Numerical Settings**
```
subsection Finite element system
  set Polynomial degree = 1             # Linear elements
  set Quadrature order = 2              # 2×2×2 Gauss points
end

subsection Time
  set Delta de = 5e-4                   # Displacement increment
  set load_rate = 0.0165                # Loading rate (mm/s)
end
```

### Parameter Sensitivity

| Parameter | Physical Meaning | Typical Range | Effect |
|-----------|------------------|---------------|---------|
| `mu1`, `mu2` | Network stiffness | 100-10000 Pa | Overall stiffness |
| `gamma_dot_0` | Viscous rate | 1e8-1e12 s⁻¹ | Relaxation speed |
| `sigma0` | Yield threshold | 1-50 Pa | Plastic onset |
| `wnp` | Nanoparticle fraction | 0-0.1 | Reinforcement |
| `zita` | Moisture content | 0-0.05 | Softening effect |

## Output

### Generated Files

1. **`solution-*.vtk`**: Paraview visualization files
   - Displacement fields
   - Stress components (Cauchy stress)
   - Strain measures (Green-Lagrange)
   - Damage variables

2. **`data-for-gnuplot.sol`**: Time history data
   - Force-displacement curves
   - Volume evolution
   - Reaction forces at boundaries

3. **`data_times.txt`**: Performance metrics
   - Assembly times
   - Solution times
   - MPI communication overhead

### Visualization

**Paraview workflow:**
1. Load `solution-*.vtk` files
2. Apply "Warp by Vector" filter for deformed configuration
3. Color by stress components or damage
4. Create animations for loading cycles

**Gnuplot example:**
```gnuplot
plot 'data-for-gnuplot.sol' using 6:9 with lines title 'Force-Displacement'
```

## Examples

### Example 1: Pure Epoxy Matrix
```
# In parameters.prm
set wnp = 0.0          # No nanoparticles
set zita = 0.0         # Dry conditions
set mu1 = 800.0        # Base matrix properties
```

### Example 2: Nanocomposite (5% CNT)
```
set wnp = 0.05         # 5% nanoparticle weight fraction
set zita = 0.0         # Dry conditions
# Networks automatically amplified by factor X
```

### Example 3: Moisture Effects
```
set wnp = 0.0          # Pure matrix
set zita = 0.02        # 2% moisture content
# Softening through α_Z factor
```

### Example 4: ML Acceleration
```
set Switch to ML = On
set At which timestep switch to ML ? = 200
# Hybrid physics→ML transition
```

## Contributing

We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch
3. **Test** your changes thoroughly
4. **Submit** a pull request with detailed description

### Development Setup

```bash
# Debug build for development
cmake -DDEAL_II_DIR=/path/to/deal.II -DCMAKE_BUILD_TYPE=Debug .
make
```

### Testing

```bash
# Run with small problem for testing
# Modify parameters.prm for quick convergence
./vevp_ml_model
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{bahtiri2023machine,
  title={A machine learning-based viscoelastic--viscoplastic model for epoxy nanocomposites with moisture content},
  author={Bahtiri, Betim and Arash, Behrouz and Scheffler, Sven and Jux, Maximilian and Rolfes, Raimund},
  journal={Computer Methods in Applied Mechanics and Engineering},
  volume={415},
  pages={116293},
  year={2023},
  publisher={Elsevier},
  doi={10.1016/j.cma.2023.116293}
}
```

## License

This project is licensed under the GNU General Public License v2.1 - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **deal.II Development Team** for the finite element library
- **Leibniz Universität Hannover** for research support
- **Original step-44 tutorial** authors for the foundation code

## Contact

For questions or support:
- **Lead Developer**: Betim Bahtiri
- **Institution**: Leibniz Universität Hannover
- **Email**: [contact information]
- **Issues**: Use GitHub issue tracker for bug reports and feature requests

---

**Note**: This implementation is designed for research purposes. For production use, please validate results against experimental data and consider additional verification steps.
