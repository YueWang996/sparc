# Asset Files Documentation

This directory contains robot model files in two different formats for different purposes.

## File Overview

### URDF Files (for bard-pytorch-dynamics library)

URDF (Unified Robot Description Format) files are used with the bard-pytorch-dynamics library for solving robot dynamics problems.

#### `spine.urdf`
- **Purpose**: Defines the spine unit as a standalone component
- **Usage**: Use this file when you need to simulate or analyze the spine mechanism in isolation
- **Library**: bard-pytorch-dynamics

#### `spined_dog.urdf`
- **Purpose**: Defines a quadruped dog robot with the spine integrated into its body
- **Usage**: Use this file for full-body dynamics simulation with the spine attached to the dog's body
- **Library**: bard-pytorch-dynamics
- **Note**: This model includes full inertia properties for all body parts

#### `spined_dog_spine_dyn.urdf`
- **Purpose**: Defines a quadruped dog robot with the spine integrated, but with simplified leg dynamics
- **Usage**: Use this file when you want to focus on spine dynamics while ignoring leg inertia for computational simplicity
- **Library**: bard-pytorch-dynamics
- **Key Difference**: The spine is attached to the main body of the dog (which retains full inertia since the body is heavy), but the leg inertia is ignored for simplification

### XML Files (for MuJoCo simulation)

XML files are used for MuJoCo (Multi-Joint dynamics with Contact) physics simulation engine.

#### `spine.xml`
- **Purpose**: Defines the spine unit as a standalone component for MuJoCo simulation
- **Usage**: Use this file when simulating the spine mechanism in MuJoCo
- **Physics Engine**: MuJoCo

#### `spined_dog.xml`
- **Purpose**: Defines a quadruped dog robot with the spine integrated for MuJoCo simulation
- **Usage**: Use this file for full-body physics simulation of the spined dog in MuJoCo
- **Physics Engine**: MuJoCo

## Summary Table

| File | Format | Model Type | Purpose | Library/Engine |
|------|--------|-----------|---------|----------------|
| `spine.urdf` | URDF | Spine only | Spine dynamics analysis | bard-pytorch-dynamics |
| `spined_dog.urdf` | URDF | Full dog + spine | Full dynamics with complete inertia | bard-pytorch-dynamics |
| `spined_dog_spine_dyn.urdf` | URDF | Dog + spine (simplified) | Spine-focused dynamics with simplified legs | bard-pytorch-dynamics |
| `spine.xml` | XML | Spine only | Spine MuJoCo simulation | MuJoCo |
| `spined_dog.xml` | XML | Full dog + spine | Full MuJoCo simulation | MuJoCo |
