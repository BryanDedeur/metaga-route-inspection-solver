# MetaGA Route Inspection Solver

The research source code for a Metaheuristic Genetic Algorithm (MetaGA) that solves route inspection arc routing problems min max k chinese postman problem (mmkcpp) and multi depot variants. 

This repository uses [PyGAD](https://github.com/ahmedfgad/GeneticAlgorithmPython) for easy to adopt source code and genetic algorithm parameters for further experimentation and [Weights & Biases](https://wandb.ai/home) for experiment tracking and dataset versioning.

## Experiments

### Runs / Random Seeds

Here is a list of 30 random seeds used in my experiments.

```
8115,3520,8647,9420,3116,6377,6207,4187,3641,8591,3580,8524,2650,2811,9963,7537,3472,3714,8158,7284,6948,6119,5253,5134,7350,2652,9968,3914,6899,4715
```
 
## Getting Started

Add bridge graph instances
```
git submodule add https://github.com/BryanDedeur/bridge-graph-instances
```

### Running 
Example

```bash
python main.py -i benchmarks/bmcv/C05.dat -k 0,0 -s 8115,3520,8647,9420,3116,6377 -j MMMR
```

### Windows & VSCode Environment

It is recommended to use Anaconda to create a isolated environment with the necessary packages versions that support PyGAD since the various dependent libraries (numpy, scipy) .

1. Install [Anaconda](https://www.anaconda.com/products/distribution)
2. Run Anaconda
3. Add Python environment version [3.X.X] that supports PyGAD (see [pygad latest docs](https://pygad.readthedocs.io/en/latest/))
4. Open Terminal through Anaconda for the Environment
5. Add pip packages
    - `pip install Numpy` - *for dependencies in pygad*
    - `pip install PyGAD` - *for genetic algorithm*
    - `pip install wandb` - *for database*
6. Edit vscode python interpreter
    - `CTRL + SHIFT + P`
    - Type `python interpreter`
    - Select the conda environment 
7. [Optional] Edit environment variables to support wandb



    
