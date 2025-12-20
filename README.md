<div align="center">
  
# [Datasets-Features-Compatibility.](https://github.com/BrenoFariasdaSilva/Datasets-Features-Compatibility) <img src="https://github.com/devicons/devicon/blob/master/icons/python/python-original.svg"  width="3%" height="3%">

</div>

<div align="center">
  
---

Welcome to the [Datasets-Features-Compatibility](https://github.com/BrenoFariasdaSilva/Datasets-Features-Compatibility) repository — a toolkit designed to inspect and describe tabular datasets (CSV) at scale, with a focus on feature compatibility both within and across datasets.  

This tool is ideal for data scientists and machine learning practitioners who need to quickly assess dataset structure and compatibility, enabling efficient analysis and research across multiple datasets with shared or overlapping features.  

---

</div>

<div align="center">

![GitHub Code Size in Bytes](https://img.shields.io/github/languages/code-size/BrenoFariasdaSilva/Datasets-Features-Compatibility)
![GitHub Commits](https://img.shields.io/github/commit-activity/t/BrenoFariasDaSilva/Datasets-Features-Compatibility/main)
![GitHub Last Commit](https://img.shields.io/github/last-commit/BrenoFariasdaSilva/Datasets-Features-Compatibility)
![GitHub Forks](https://img.shields.io/github/forks/BrenoFariasDaSilva/Datasets-Features-Compatibility)
![GitHub Language Count](https://img.shields.io/github/languages/count/BrenoFariasDaSilva/Datasets-Features-Compatibility)
![GitHub License](https://img.shields.io/github/license/BrenoFariasdaSilva/Datasets-Features-Compatibility)
![GitHub Stars](https://img.shields.io/github/stars/BrenoFariasdaSilva/Datasets-Features-Compatibility)
![wakatime](https://wakatime.com/badge/github/BrenoFariasdaSilva/Datasets-Features-Compatibility.svg)

</div>

<div align="center">
  
![RepoBeats Statistics](https://repobeats.axiom.co/api/embed/151aaa14958b56988adc39f88605c450efb671f0.svg "Repobeats analytics image")

</div>

## Table of Contents
- [Datasets-Features-Compatibility. ](#datasets-features-compatibility-)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Setup](#setup)
    - [Git](#git)
        - [Linux](#linux)
        - [macOS](#macos)
        - [Windows](#windows)
    - [Clone the Repository](#clone-the-repository)
    - [Python, Pip and Venv](#python-pip-and-venv)
      - [Linux](#linux-1)
      - [macOS](#macos-1)
      - [Windows](#windows-1)
    - [Make](#make)
      - [Linux](#linux-2)
      - [macOS](#macos-2)
      - [Windows](#windows-2)
    - [Dependencies](#dependencies)
  - [Usage/Run](#usagerun)
  - [Results](#results)
  - [How to Cite?](#how-to-cite)
  - [Contributing](#contributing)
  - [Collaborators](#collaborators)
  - [License](#license)
    - [Apache License 2.0](#apache-license-20)

## Introduction

This repository provides a toolkit to inspect and describe tabular datasets (CSV) at scale, with a focus on feature compatibility and dataset comparability. The main goals are:

- Recursively discover dataset files and build per-file metadata (sample and feature counts, data types, missing-value summaries, and detected label columns).
- Produce per-dataset descriptor CSVs that list common and extra features, class distributions, and basic dataset statistics.
- Perform pairwise cross-dataset comparisons to identify shared and non-shared features across dataset collections.
- Optionally produce t-SNE visualizations of numeric feature spaces for quick separability checks.

Typical uses:

- Data curation and quality checks before model training.
- Dataset compatibility analysis when combining multiple sources or performing transfer learning experiments.
- Quick exploratory summaries for large datasets where full analysis would be costly.

Outputs are written into a `Dataset_Description` folder inside each dataset path and include `Dataset_Descriptor.csv` (per-dataset) and `Cross_Dataset_Descriptor.csv` for pairwise comparisons.

## Setup

This section provides instructions for installing Git, Python, Pip, Make, then to clone the repository (if not done yet) and all required project dependencies. 

### Git

`git` is a distributed version control system that is widely used for tracking changes in source code during software development. In this project, `git` is used to download and manage the analyzed repositories, as well as to clone the project and its submodules. To install `git`, follow the instructions below based on your operating system:

##### Linux

To install `git` on Linux, run:

```bash
sudo apt install git -y
```

##### macOS

To install `git` on MacOS, you can use Homebrew:

```bash
brew install git
```

##### Windows

On Windows, you can download `git` from the official website [here](https://git-scm.com/downloads) and follow the installation instructions provided there.

### Clone the Repository

Now that git is installed, it's time to clone this repository with all required submodules, use:

``` bash
git clone --recurse-submodules https://github.com/BrenoFariasdaSilva/Datasets-Features-Compatibility.git
```

If you clone without submodules (not recommended):

``` bash
git clone https://github.com/BrenoFariasdaSilva/Datasets-Features-Compatibility
```

To initialize submodules manually:

``` bash
cd Datasets-Features-Compatibility # Only if not in the repository root directory yet
git submodule init
git submodule update
```

### Python, Pip and Venv

You must have Python 3, Pip, and the `venv` module installed.

#### Linux

``` bash
sudo apt install python3 python3-pip python3-venv -y
```

#### macOS

``` bash
brew install python3
```

#### Windows

If you do not have Chocolatey installed, you can install it by running the following command in an **elevated PowerShell (Run as Administrator)**:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))
```

Once Chocolatey is installed, you can install Python using:

``` bash
choco install python3
```

Or download the installer from the official Python website.

### Make 

`Make` is used to run automated tasks defined in the project's Makefile, such as setting up environments, executing scripts, and managing Python dependencies.

#### Linux

``` bash
sudo apt install make -y
```

#### macOS

``` bash
brew install make
```

#### Windows

Available via Cygwin, MSYS2, or WSL.

### Dependencies

1. Install the project dependencies with the following command:

   ```bash
   cd Datasets-Features-Compatibility # Only if not in the repository root directory yet
   make dependencies
   ```

## Usage/Run

Before running the project, change to the repository root directory:

```bash
cd Datasets-Features-Compatibility
```

Create the datasets directory (if you don't already have one):

```bash
mkdir ./Datasets
```

This project expects you to organize datasets according to the `DATASETS` mapping defined in `main.py`. Example (excerpt from `main.py`):

```python
DATASETS = { # Dictionary containing dataset paths and feature files
	"CICDDoS2019-Dataset": [ # List of paths to the CICDDoS2019 dataset
		"./Datasets/CICDDoS2019/01-12/",
   ],
   "CICDDoS2017-Dataset": [ # List of paths to the CICDDoS2017 dataset
      "./Datasets/CIC-IDS2017/Converted/",
   ],
}

```

How `main.py` behaves:

1. It uses the `DATASETS` mapping to find dataset folders and files to analyze.
2. It recursively scans the provided dataset paths for supported file format: `.csv`. In case your datasets are in other formats, i'm happy to say there i've created a repository that reads any `.arff`, `.csv`, `.parquet`, and `.txt` and converts them to all those formats as well. You can find this repo in [Multi-Format-Dataset-Converter](Multi-Format-Dataset-Converter).
 
Note about paths and file types:

- The values in the `DATASETS` mapping may be either directory paths (the script will scan them recursively) or direct file paths to individual CSV files. Example:

```python
DATASETS = {
   "CICDDoS2019-Dataset": ["./Datasets/CICDDoS2019/01-12/test.csv"],
   "CICDDoS2017-Dataset": ["./Datasets/CIC-IDS2017/Converted/example.csv"],
}
```

- All provided files must be CSV files (the script's current loader expects CSV input). If you need other formats, consider converting them to CSV before analysis or use the external converter linked above.
3. It performs dataset inspection and feature analysis (header detection, feature types, missing values, class distributions, etc.).
4. If `CROSS_DATASET_VALIDATE` is `True`, it performs cross-dataset header/feature comparisons across the defined datasets, otherwise only for the files in the same dataset.
5. The consolidated results are written to `RESULTS_DIR/RESULTS_FILENAME` (by default `./Dataset_Description/Dataset_Descriptor.csv` in the dir of the current dataset processed).
6. Files and directories matching `IGNORE_FILES` and `IGNORE_DIRS` are skipped during scanning.

Configurable constants you may want to edit in `main.py`:

```python
# Execution Constants (examples):
VERBOSE = False # Set to True to output verbose messages
DATASETS = { ... } # Dataset mapping shown above
CROSS_DATASET_VALIDATE = True # Set to True to perform cross-dataset validation between the datasets defined in DATASETS

RESULTS_DIR = "./Dataset_Description/" # Directory to save the results
RESULTS_FILENAME = "Dataset_Descriptor.csv" # Filename for the results CSV

IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"] # List of directory names to ignore when searching for datasets
```

Run the project (the repository Makefile prepares the environment and runs `main.py`):

```bash
make
```

## Results

After running the script with the following configuration in `main.py`:

```python
DATASETS = { # Dictionary containing dataset paths and feature files
	"CICDDoS2019-Dataset": [ # List of paths to the CICDDoS2019 dataset
		"./Datasets/CICDDoS2019/01-12/",
   ],
   "CICDDoS2017-Dataset": [ # List of paths to the CICDDoS2017 dataset
      "./Datasets/CIC-IDS2017/Converted/",
   ],
}

CROSS_DATASET_VALIDATE = True # Set to True to perform cross-dataset validation between the datasets defined in DATASETS

RESULTS_DIR = "./Dataset_Description/" # Directory to save the results
RESULTS_FILENAME = "Dataset_Descriptor.csv" # Filename for the results CSV

IGNORE_FILES = [RESULTS_FILENAME] # List of filenames to ignore when searching for datasets
IGNORE_DIRS = ["Cache", "Data_Separability", "Dataset_Description", "Feature_Analysis"] # List of directory names to ignore when searching for datasets
```

The run produced per-dataset descriptor reports and cross-dataset comparison reports. Key generated outputs (examples):

- Per-dataset descriptors:
  - `Datasets/CICDDoS2019/01-12/Dataset_Description/Dataset_descriptor.csv` — a row per discovered file containing: dataset name, size, sample & feature counts, feature types, categorical columns, missing values summary, classes and class distributions, whether headers match across files, common/extra features, and t-SNE plot path.
  - `Datasets/CIC-IDS2017/Converted/Dataset_Description/Dataset_Descriptor.csv` — same format for the CIC-IDS2017 files.

- Cross-dataset descriptors (pairwise comparisons):
  - `Datasets/CICDDoS2019/01-12/Dataset_Description/Cross_Dataset_Descriptor.csv` — shows comparison between `CICDDoS2019-Dataset` and `CICDDoS2017-Dataset` (example: `Files in A = 18`, `Files in B = 8`, lists of common and extra features).
  - `Datasets/CIC-IDS2017/Converted/Dataset_Description/Cross_Dataset_Descriptor.csv` — the reciprocal comparison (same content from the other dataset folder view).

Example details observed in this run:

- The CICDDoS2019 per-dataset descriptor reported multiple large CSVs (TFTP: ~19.5M rows, various DrDoS_* files with millions of rows). Many files had 76 features with numeric and integer types and an identified `label` column; headers were reported as matching across files where applicable.
- The CIC-IDS2017 per-dataset descriptor reported several files (~68–70 features) with class distributions (e.g., `Benign`, `Bot`, `DDoS`, etc.).

Notes and where to look:

- Results are written per dataset into a `Dataset_Description` folder inside each dataset path (e.g. `Datasets/<dataset_path>/Dataset_Description/`).
- The main consolidated filenames are `Dataset_Descriptor.csv` (per-dataset) and `Cross_Dataset_Descriptor.csv` (pairwise comparisons).
- Open the generated CSVs with a spreadsheet viewer or `pandas` to inspect feature lists and distributions; t-SNE plots (if generated) are saved alongside the descriptors.

If you want, I can add a short script or Make target to collect all per-dataset `Dataset_Descriptor.csv` files into a single top-level summary file.

## How to Cite?

If you use the Datasets-Features-Compatibility in your research, please cite it using the following BibTeX entry:

```
@misc{softwareDatasets-Features-Compatibility:2025,
  title = {Datasets-Features-Compatibility: Project-Description},
  author = {Breno Farias da Silva},
  year = {2025},
  howpublished = {https://github.com/BrenoFariasdaSilva/Datasets-Features-Compatibility},
  note = {Accessed on December 12, 2025}
}
```

Additionally, a `main.bib` file is available in the root directory of this repository, in which contains the BibTeX entry for this project.

If you find this repository valuable, please don't forget to give it a ⭐ to show your support! Contributions are highly encouraged, whether by creating issues for feedback or submitting pull requests (PRs) to improve the project. For details on how to contribute, please refer to the [Contributing](#contributing) section below.

Thank you for your support and for recognizing the contribution of this tool to your work!

## Contributing

Contributions are what make the open-source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**. If you have suggestions for improving the code, your insights will be highly welcome.
In order to contribute to this project, please follow the guidelines below or read the [CONTRIBUTING.md](CONTRIBUTING.md) file for more details on how to contribute to this project, as it contains information about the commit standards and the entire pull request process.
Please follow these guidelines to make your contributions smooth and effective:

1. **Set Up Your Environment**: Ensure you've followed the setup instructions in the [Setup](#setup) section to prepare your development environment.

2. **Make Your Changes**:
   - **Create a Branch**: `git checkout -b feature/YourFeatureName`
   - **Implement Your Changes**: Make sure to test your changes thoroughly.
   - **Commit Your Changes**: Use clear commit messages, for example:
     - For new features: `git commit -m "FEAT: Add some AmazingFeature"`
     - For bug fixes: `git commit -m "FIX: Resolve Issue #123"`
     - For documentation: `git commit -m "DOCS: Update README with new instructions"`
     - For refactorings: `git commit -m "REFACTOR: Enhance component for better aspect"`
     - For snapshots: `git commit -m "SNAPSHOT: Temporary commit to save the current state for later reference"`
   - See more about crafting commit messages in the [CONTRIBUTING.md](CONTRIBUTING.md) file.

3. **Submit Your Contribution**:
   - **Push Your Changes**: `git push origin feature/YourFeatureName`
   - **Open a Pull Request (PR)**: Navigate to the repository on GitHub and open a PR with a detailed description of your changes.

4. **Stay Engaged**: Respond to any feedback from the project maintainers and make necessary adjustments to your PR.

5. **Celebrate**: Once your PR is merged, celebrate your contribution to the project!

## Collaborators

We thank the following people who contributed to this project:

<table>
  <tr>
    <td align="center">
      <a href="#" title="defina o titulo do link">
        <img src="https://github.com/BrenoFariasdaSilva.png" width="100px;" alt="My Profile Picture"/><br>
        <sub>
          <b>Breno Farias da Silva</b>
        </sub>
      </a>
    </td>
  </tr>
</table>

## License

### Apache License 2.0

This project is licensed under the [Apache License 2.0](LICENSE). This license permits use, modification, distribution, and sublicense of the code for both private and commercial purposes, provided that the original copyright notice and a disclaimer of warranty are included in all copies or substantial portions of the software. It also requires a clear attribution back to the original author(s) of the repository. For more details, see the [LICENSE](LICENSE) file in this repository.
