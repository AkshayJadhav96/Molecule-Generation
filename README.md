# MoleculeGen AI

Generate, visualize, and simulate novel molecules using MoFlow and VAE models.

## Overview

MoleculeGen AI is a web-based platform for generating novel molecular structures using generative machine learning models, specifically MoFlow and Variational Autoencoders (VAE). The system allows users to generate molecules, analyze their properties (e.g., QED, SA scores, rotatable bonds), and visualize the creation process through animated simulations. This project combines a Flask backend for model inference and property computation with a dynamic frontend for an interactive user experience.

## Key Features

- Generate molecules using MoFlow and VAE models with customizable parameters
- Visualize molecules in 2D with RDKit-generated images
- Simulate molecule creation with a 150-frame GIF animation (6 phases: latent sampling, atom placement, bond connection, relaxation, atom typing, final display)
- Analyze molecular properties like QED, SA scores, molecular weight, logP, and rotatable bonds
- User-friendly interface with sliders, pagination, and export/save functionality

## Project Setup

### Prerequisites

- Python 3.8+: Ensure Python is installed on your system
- UV: A Python package and dependency manager (similar to pip but faster). Install UV by following the instructions at [uv.rs](https://uv.rs)
- Git: To clone the repository

### Installation Steps

1. Clone the Repository:
   ```bash
   git clone https://github.com/AkshayJadhav96/Molecule-Generation.git
   ```

2. Navigate to the Project Directory:
   ```bash
   cd Molecule-Generation
   ```

3. Sync Dependencies with UV:
   ```bash
   uv sync
   ```

### Running the Website

To launch the web application, use the following command from the project root directory:
```bash
uv run ui/app.py
```

This will start the Flask server, and the website will be accessible at http://127.0.0.1:5000/. Open this URL in your browser to interact with the platform.

**Note**: All commands (e.g., training, generation) should be run using `uv run` to ensure the correct environment and dependencies are used. For example:
- To run a script like generate.py: `uv run python generate.py`

## Project Structure

The repository is organized into several key directories and files:

- **models/**: Contains model-related code for MoFlow, sourced from calvin-zcx/moflow with modifications to fit our project requirements.
- **utils/**: Utility functions for MoFlow, also adapted from the same MoFlow repository.
- **MoFlow Scripts** (adapted from calvin-zcx/moflow):
  - `run_train.py`: Script to initiate MoFlow training.
  - `train_model.py`: Core training logic for MoFlow.
  - `train_moflow.py`: Specific training functions for MoFlow.
  - `generate.py`: Generates molecules using the trained MoFlow model.
  - `optimize_property.py`: Optimizes molecular properties (e.g., QED) during generation.
  - **Modifications**: We made changes to these scripts to integrate with our Flask backend, support the QM9 dataset, and add temperature-based generation control.
- **sascorer.py**: Sourced from the web to calculate Synthetic Accessibility (SA) scores for molecules. Integrated into our property computation pipeline.
- **ui/**: Contains all website-related files:
  - `app.py`: Flask backend handling API endpoints (/generate, /simulate, etc.).
  - Frontend files (HTML, CSS, JavaScript) for the web interface, including main.js for interactivity.
- **VAE/**: Houses all VAE model-related files, including model architecture, training scripts, and inference logic.
- **results/**: Stores trained model weights and checkpoints for both MoFlow and VAE models.

## Dependencies

The project relies on several libraries, installed via `uv sync`:

- **PyTorch**: For training and inference of MoFlow and VAE models
- **RDKit**: For SMILES validation, property computation (e.g., QED, rotatable bonds), and molecule visualization
- **rdkit-contrib**: For calculating SA scores using sascorer.py
- **Flask**: Backend web framework
- **Matplotlib**: For creating GIF animations in the simulation feature
- **NumPy**: For numerical operations
- **Tailwind CSS & Font Awesome**: For frontend styling and icons

## Usage

### Generate Molecules
1. Navigate to http://127.0.0.1:5000/ after starting the server
2. Select a model (MoFlow or VAE), adjust parameters (e.g., temperature), and click "Generate Molecules"
3. View generated molecules with their properties (e.g., QED, SA, rotatable bonds)

### Simulate Molecule Creation
- Click the "Simulate" button on any generated molecule to see a 6-phase animation of its creation process

### Analyze and Export
- Use the "Analyze" button to explore molecule details
- Export or save molecules for further use

## Data Sources

- **QM9 Dataset**: Used as the primary dataset, containing ~134,000 small organic molecules with up to 9 heavy atoms (C, O, N, F). Includes SMILES strings and quantum mechanical properties.
- **ZINC Database**: Supplementary dataset with over 250,000 compounds for validation and diversity.

## Models

### MoFlow
- **Source**: Adapted from calvin-zcx/moflow
- **Customizations**: Modified training and generation scripts to support QM9 data, integrate with Flask, and add temperature control
- **Functionality**: Generates diverse molecules by modeling the joint distribution of molecular graphs using normalizing flows

### VAE
- **Implementation**: Custom-built using PyTorch, located in the VAE/ directory
- **Functionality**: Encodes SMILES into a latent space and decodes them to generate new molecules, offering better control over generation

## Results

- **Generation**: MoFlow achieves ~90% validity and a diversity score of ~0.8; VAE achieves ~85% validity with more controlled outputs
- **Properties**: QED (0.5-0.9), SA scores (1.5-6), and rotatable bonds (0-5) align with drug-like criteria
- **Simulation**: Smooth 150-frame GIFs for molecules up to 20 atoms

Trained models are stored in the `results/` directory for both MoFlow and VAE.

## Acknowledgments

- We express gratitude to [calvin-zcx/moflow](https://github.com/calvin-zcx/moflow) for providing the foundational MoFlow code, which we adapted for our project. Specifically, the models/, utils/, data/, and scripts like run_train.py, train_model.py, train_moflow.py, generate.py, and optimize_property.py were sourced from their repository and modified to suit our needs.
- The sascorer.py script for calculating Synthetic Accessibility (SA) scores was sourced from the web and integrated into our pipeline.
- Thanks to the open-source community for tools like RDKit, PyTorch, and Flask, which made this project possible.