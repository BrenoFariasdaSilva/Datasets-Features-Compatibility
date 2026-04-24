"""
================================================================================
Dataset Descriptor and Report Generator - dataset_descriptor.py
================================================================================
Author      : Breno Farias da Silva
Created     : 2025-10-07

What this module does
    - Recursively scans a directory (or single CSV file) and collects all
        matching CSV datasets.
    - Extracts metadata and summaries per file: sample/feature counts,
        feature types, missing values, detected label column and class
        distributions.
    - Optionally generates a 2D t-SNE plot per file (`Data_Separability/`),
        using class-aware downsampling with a default target of 2000 (config
        in callers); small classes (default min 50) are preserved in full.
    - Optionally computes cross-dataset compatibility reports comparing
        feature unions/intersections between dataset groups (`CROSS_DATASET_VALIDATE`).

Key defaults and globals
    - File discovery default extension: .csv
    - Results saved under each dataset base directory in `RESULTS_DIR`
        (default: ./Dataset_Description/). The per-dataset CSV is named
        `Dataset_Descriptor.csv` (config: `RESULTS_FILENAME`).
    - Cross-group report: saved as `Cross_{RESULTS_FILENAME}` in each
        group's results directory when `CROSS_DATASET_VALIDATE = True`.
    - t-SNE: uses sklearn.manifold.TSNE and adapts to `n_iter`/`max_iter`
        parameter name differences across scikit-learn versions.

Behavioral notes & guarantees
    - Downsampling is class-aware: classes with >= `min_class_size` receive
        at least `min_class_size` samples when possible; classes with fewer
        samples are included entirely. Remaining budget is distributed
        proportionally using a fractional remainder method.
    - Numeric extraction tries `select_dtypes(include=["number"])` and
        attempts coercion of object/string columns to numeric when needed.
    - The script performs disk-space verification before writing large outputs.
    - The generator writes one cross-dataset CSV per dataset group and
        normalizes rows so the file's group appears as "Dataset A".

Usage
    - Run the script directly: `python3 dataset_descriptor.py` (adjust
        `DATASETS` constant or call `generate_dataset_report()` programmatically).

Dependencies
    - Python 3.9+
    - pandas, numpy, matplotlib, scikit-learn, tqdm, colorama

Limitations / TODO
    - Header detection and CSV parsing are pragmatic; malformed CSVs may
        require preprocessing.
    - Add CLI flags for `sample_size`, `min_class_size`, `CROSS_DATASET_VALIDATE`.
    - Consider structured logging instead of printing/redirecting stdout.
"""


import argparse  # For parsing CLI arguments
import atexit  # For playing a sound when the program finishes
import dataframe_image as dfi  # For exporting DataFrame as PNG images
import datetime  # For timestamping
import gc  # For explicit garbage collection
import matplotlib.pyplot as plt  # For plotting t-SNE results
import multiprocessing as mp  # For explicit process and semaphore resource finalization
import numpy as np  # For numerical operations
import os  # For running a command in the terminal
import pandas as pd  # For data manipulation
import platform  # For getting the operating system name
import re  # For regex operations
import sys  # For system-specific parameters and functions
import time  # Import time locally to perform retry timing and ensure dependency is available at runtime
import traceback  # For printing full exception tracebacks
import warnings  # For suppressing pandas warnings when requested
import yaml  # For optional config.yaml loading when locating WGANGP outputs
from colorama import Style  # For coloring the terminal
from inspect import signature  # For inspecting function signatures
from Logger import Logger  # For logging output to both terminal and file
from mpl_toolkits.mplot3d import Axes3D  # For 3D plotting
from pathlib import Path  # For handling file paths
from PIL import Image  # For verifying image dimensions and upscaling if necessary
from sklearn.manifold import TSNE  # For t-SNE dimensionality reduction
from sklearn.preprocessing import StandardScaler  # For feature scaling
from tqdm import tqdm  # For progress bars
from typing import Any, cast  # For type hinting


# Macros:
class BackgroundColors:  # Colors for the terminal
    CYAN = "\033[96m"  # Cyan
    GREEN = "\033[92m"  # Green
    YELLOW = "\033[93m"  # Yellow
    RED = "\033[91m"  # Red
    BOLD = "\033[1m"  # Bold
    UNDERLINE = "\033[4m"  # Underline
    CLEAR_TERMINAL = "\033[H\033[J"  # Clear the terminal


# Execution Constants will be sourced from configuration (CLI > config.yaml > defaults)


SOUND_COMMANDS = {
    "Darwin": "afplay",
    "Linux": "aplay",
    "Windows": "start",
}

SOUND_FILE = "./.assets/Sounds/NotificationSound.wav"


# Functions Definitions:


def main():
    """
    Main function.

    :return: None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring

        cli_args_dict = parse_cli_args()  # Parse CLI arguments and load configuration as dict
        config = get_config(file_path=cli_args_dict.get("config", "config.yaml"), cli_args=cli_args_dict)  # Load and merge config with CLI overrides

        cli_args_ns = argparse.Namespace(**cli_args_dict)  # Convert dict to Namespace for type safety

        runtime = init_runtime(config)  # Initialize runtime artifacts including the logger

        sys_stdout_old = sys.stdout  # Save original stdout for later restoration
        sys_stderr_old = sys.stderr  # Save original stderr for later restoration
        sys.stdout = runtime["logger"]  # Redirect stdout to logger for this runtime session
        sys.stderr = runtime["logger"]  # Redirect stderr to logger for this runtime session

        os.environ["DD_DESCRIPTOR_VERBOSE"] = str(runtime.get("verbose", False))  # Export verbosity flag via environment for use in other functions

        print(f"{BackgroundColors.CLEAR_TERMINAL}{BackgroundColors.BOLD}{BackgroundColors.GREEN}Welcome to the {BackgroundColors.CYAN}Dataset Descriptor{BackgroundColors.GREEN}!{Style.RESET_ALL}")  # Print welcome message
        start_time = datetime.datetime.now()  # Capture program start time

        log_config_sources(config, cli_args_dict)  # Log resolved configuration values with their source
        low_memory = resolve_low_memory(cli_args_ns, config)  # Determine low memory mode based on CLI and config settings

        datasets = config.get("dataset_descriptor", {}).get("datasets", {}) or config.get("datasets") or {}  # Resolve datasets mapping from config
        results_suffix = config.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")  # Resolve output CSV suffix from config

        for dataset_name, paths in datasets.items():  # Iterate over configured dataset entries
            dataset_name = str(dataset_name).strip()  # Normalize dataset name by removing leading/trailing spaces
            verbose_output(f"{BackgroundColors.BOLD}{BackgroundColors.GREEN}Processing dataset: {BackgroundColors.CYAN}{dataset_name}{Style.RESET_ALL}")  # Log dataset processing start
            safe_dataset_name = dataset_name.replace(" ", "_").replace("/", "_")  # Sanitize dataset name for safe filesystem use and remove leading/trailing spaces

            for dir_path in paths:  # Iterate over all configured paths for this dataset
                dir_path = str(dir_path).strip()  # Normalize directory path by removing leading/trailing spaces
                print(f"{BackgroundColors.GREEN}Location: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Print current directory path
                if not verify_filepath_exists(dir_path):  # Verify the configured path exists before processing
                    print(f"{BackgroundColors.RED}The specified input path does not exist: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Report missing path to terminal
                    continue  # Skip non-existing paths without aborting the full run

                success = generate_dataset_report(dir_path, file_extension=".csv", low_memory=low_memory, output_filename=None, config=config)  # Generate dataset report for this path
                if not success:  # Verify whether report generation succeeded
                    print(f"{BackgroundColors.RED}Failed to generate dataset report for: {BackgroundColors.CYAN}{dir_path}{Style.RESET_ALL}")  # Report failure for this path
                else:  # Report generation succeeded
                    print(f"{BackgroundColors.GREEN}Report saved for {BackgroundColors.CYAN}{safe_dataset_name}{BackgroundColors.GREEN} -> {BackgroundColors.CYAN}{results_suffix}{Style.RESET_ALL}")  # Confirm successful report save

        if config.get("execution", {}).get("cross_dataset_validate", True) and len(datasets) > 1:  # Verify cross-dataset validation is enabled and multiple datasets are configured
            try:  # Attempt cross-dataset validation with graceful failure handling
                success = generate_cross_dataset_report(datasets, file_extension=".csv", low_memory=low_memory, config=config)  # Generate pairwise cross-dataset feature compatibility report
                if success:  # Verify whether cross-dataset report was saved
                    print(f"{BackgroundColors.GREEN}Cross-dataset report saved -> {BackgroundColors.CYAN}Cross_{results_suffix.lstrip('_')}{Style.RESET_ALL}")  # Confirm successful cross-dataset report save
                else:  # Cross-dataset report generation produced no output
                    print(f"{BackgroundColors.YELLOW}No cross-dataset comparisons generated (no files found).{Style.RESET_ALL}")  # Warn when no output was produced
            except Exception as e:  # Catch cross-dataset validation errors to avoid aborting the main run
                print(f"{BackgroundColors.RED}Cross-dataset validation failed: {e}{Style.RESET_ALL}")  # Report cross-dataset validation failure without re-raising

        sys.stdout = sys_stdout_old  # Restore original stdout after logging session
        sys.stderr = sys_stderr_old  # Restore original stderr after logging session

        finish_time = datetime.datetime.now()  # Get the finish time of the program
        print(
            f"{BackgroundColors.GREEN}Start time: {BackgroundColors.CYAN}{start_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Finish time: {BackgroundColors.CYAN}{finish_time.strftime('%d/%m/%Y - %H:%M:%S')}\n{BackgroundColors.GREEN}Execution time: {BackgroundColors.CYAN}{calculate_execution_time(start_time, finish_time)}{Style.RESET_ALL}"
        )  # Output the start and finish times
        print(
            f"\n{BackgroundColors.BOLD}{BackgroundColors.GREEN}Program finished.{Style.RESET_ALL}"
        )  # Output the end of the program message

        try:  # Attempt to register sound notification at exit when configured
            if config.get("execution", {}).get("play_sound", True):  # Verify play_sound is enabled in config
                atexit.register(play_sound)  # Register play_sound to execute when the program exits
        except Exception:  # Ignore any errors during atexit registration to avoid crashing at exit
            pass  # Continue silently when atexit registration fails

        finalize_multiprocessing_resources()  # Finalize child processes and tracked shared resources before interpreter shutdown
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


if __name__ == "__main__":
    """
    This is the standard boilerplate that calls the main() function.

    :return: None
    """

    try:  # Protect main execution to ensure errors are reported and notified
        configure_multiprocessing_startup()  # Configure multiprocessing start method once before executing main flow
        main()  # Call the main function
    except Exception as e:  # Catch any unhandled exception from main
        print(str(e))  # Print the exception message to terminal for logs
        raise  # Re-raise to avoid silent failure and preserve original crash behavior
