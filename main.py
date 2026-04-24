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


def collect_group_files(paths, file_extension=".csv", config: dict | None = None):
    """
    Collect all matching files for a group of paths.

    :param paths: List of file or directory paths to search.
    :param file_extension: File extension to filter (default: ".csv").
    :param config: Optional configuration dictionary passed to collect_matching_files.
    :return: Sorted list of unique file paths.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Collecting {file_extension} files from specified paths...{Style.RESET_ALL}"
        )  # Output collection message

        files = []  # Initialize collection list

        for p in paths:  # Iterate over each path
            if os.path.isdir(p):  # If path is a directory
                files.extend(collect_matching_files(p, file_extension, config=config))  # Collect matching files
            elif os.path.isfile(p) and p.endswith(file_extension):  # If path is a file with correct extension
                files.append(p)  # Add file to list

        unique_files = list(set(files))  # Remove duplicates while preserving no particular order

        files_with_size = []  # Prepare list to hold (path, size) tuples for robust sorting
        for f in unique_files:  # Iterate files to resolve their sizes
            try:  # Attempt to get file size and handle any filesystem issues gracefully
                size = os.path.getsize(f)  # Get the file size in bytes for sorting by magnitude
            except Exception:  # If size retrieval fails for any file
                size = 0  # Fallback to zero size to avoid breaking the sort when file is inaccessible
            files_with_size.append((f, size))  # Store tuple of file path and its size for later sorting

        return [p for p, _ in sorted(files_with_size, key=lambda x: (-x[1], x[0]))]  # Sort by size descending then by path for determinism
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def compute_group_features(files, low_memory=None):
    """
    Compute common and union features for a list of dataset files.

    :param files: List of dataset file paths
    :param low_memory: Whether to optimize memory when reading CSV headers
    :return: Tuple (common_features_set, union_features_set)
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        verbose_output(
            f"{BackgroundColors.GREEN}Computing common and union features for dataset group...{Style.RESET_ALL}"
        )  # Output computation message

        if not files:  # No files, return empty sets
            return set(), set()  # Return empty sets

        headers_map = build_headers_map(files, low_memory=low_memory)  # Build headers map
        common_features, _ = compute_common_features(headers_map)  # Compute common features

        union_features = set()  # Initialize union set
        for cols in headers_map.values():  # Iterate over each file's columns
            if cols:  # If columns exist
                union_features.update(
                    [c.strip().lower() for c in cols]
                )  # Normalize features: strip whitespace and lowercase
        common_features = set(
            [c.strip().lower() for c in common_features]
        )  # Normalize common features: strip whitespace and lowercase

        return set(common_features), union_features  # Return both sets
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def generate_pairwise_report(group_info):
    """
    Generate pairwise comparison rows from group info.

    :param group_info: Dict mapping group_name -> {"files": [...], "common": set(), "union": set()}
    :return: List of dictionaries representing pairwise comparison rows
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        rows = []  # Initialize report row list
        group_names = list(group_info.keys())  # List of group names

        for i in range(len(group_names)):  # Iterate over first group
            for j in range(i + 1, len(group_names)):  # Iterate over second group avoiding duplicates
                a_name, b_name = group_names[i], group_names[j]  # Group names
                a_info, b_info = group_info[a_name], group_info[b_name]  # Group info

                if not a_info["files"] and not b_info["files"]:  # Skip if both have no files
                    continue  # Proceed to next pair

                common_between = sorted(a_info["union"] & b_info["union"])  # Features common to both groups
                extras_a = sorted(a_info["union"] - b_info["union"])  # Features in A not in B
                extras_b = sorted(b_info["union"] - a_info["union"])  # Features in B not in A

                n_common = int(len(common_between))  # Integer count of common features
                n_extra_a = int(len(extras_a))  # Integer count of extra features in A
                n_extra_b = int(len(extras_b))  # Integer count of extra features in B

                row = {  # Construct row dictionary
                    "Dataset A": a_name,  # First dataset group name
                    "Dataset B": b_name,  # Second dataset group name
                    "Files in A": len(a_info["files"]),  # Number of files in A
                    "Files in B": len(b_info["files"]),  # Number of files in B
                    "N Common Features": n_common,  # Integer count of common features between A and B
                    "Common Features (A ∩ B)": ", ".join(common_between) or "None",  # Common features between A and B
                    "N Extra Features in A": n_extra_a,  # Integer count of extra features present in A but not in B
                    "Extra Features in A (A \\ B)": ", ".join(extras_a) or "None",  # Extra features in A
                    "N Extra Features in B": n_extra_b,  # Integer count of extra features present in B but not in A
                    "Extra Features in B (B \\ A)": ", ".join(extras_b) or "None",  # Extra features in B
                }

                rows.append(row)  # Append to report rows

        return rows  # Return the list of report rows
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def adjust_rows_for_group(report_rows, group_name):
    """
    Adjust pairwise rows so that the target group always appears as Dataset A.

    :param report_rows: List of dictionaries representing pairwise report rows
    :param group_name: Target group to appear as Dataset A
    :return: List of adjusted report rows
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        adjusted = []  # Initialize adjusted row list

        for row in report_rows:  # Iterate over existing report rows
            if row["Dataset A"] == group_name:  # Already Dataset A
                adjusted.append(dict(row))  # Keep as-is
            elif row["Dataset B"] == group_name:  # Swap A <-> B
                swapped = {  # Construct swapped row
                    "Dataset A": row["Dataset B"],  # Swap Dataset A
                    "Dataset B": row["Dataset A"],  # Swap Dataset B
                    "Files in A": row["Files in B"],  # Swap file counts
                    "Files in B": row["Files in A"],  # Swap file counts
                    "N Common Features": int(row.get("N Common Features", 0)),  # Keep common feature count unchanged on swap
                    "Common Features (A ∩ B)": row["Common Features (A ∩ B)"],  # Keep common features unchanged on swap
                    "N Extra Features in A": int(row.get("N Extra Features in B", 0)),  # Swap extra feature count so A count reflects former B count
                    "Extra Features in A (A \\ B)": row["Extra Features in B (B \\ A)"],  # Swap extra features so A receives former B extras
                    "N Extra Features in B": int(row.get("N Extra Features in A", 0)),  # Swap extra feature count so B count reflects former A count
                    "Extra Features in B (B \\ A)": row["Extra Features in A (A \\ B)"],  # Swap extra features so B receives former A extras
                }
                adjusted.append(swapped)  # Append swapped row
            else:  # Unrelated row, keep as-is
                adjusted.append(dict(row))  # Keep as-is

        return adjusted  # Return adjusted rows
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def generate_cross_dataset_report(datasets_dict, file_extension=".csv", low_memory=None, output_filename=None, config: dict | None = None):
    """
    Generate a cross-dataset feature-compatibility report comparing dataset
    groups defined in `datasets_dict`. Produces pairwise comparisons between
    dataset groups and writes a CSV report named `Cross_{RESULTS_FILENAME}` by
    default into the `RESULTS_DIR`.

    :param datasets_dict: Dict mapping dataset group name -> list of paths.
    :param file_extension: Extension to search for (default: .csv).
    :param low_memory: Passed to CSV loader when building headers.
    :param output_filename: Optional filename to write; defaults to Cross_{RESULTS_FILENAME}.
    :param config: Optional configuration dictionary for resolving output paths and settings.
    :return: True on success, False otherwise.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = config or get_default_config()
        if output_filename is None:  # If no output filename is provided
            suffix = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description")  # Get suffix from config or default
            output_filename = f"Cross_{suffix.lstrip('_')}" if suffix else "Cross_dataset_descriptor.csv"  # Build cross filename
        if not output_filename.lower().endswith(".csv"):  # Verify the output filename has a .csv extension
            output_filename = f"{output_filename}.csv"  # Append .csv extension when missing

        group_info = {}  # Map group_name -> {"files": [...], "common": set(), "union": set()}
        for group_name, paths in datasets_dict.items():  # Iterate over dataset groups
            all_files = collect_group_files(paths, file_extension, config=cfg)  # Collect files for this group
            common_features, union_features = compute_group_features(all_files, low_memory=low_memory)  # Compute features

            group_info[group_name] = {
                "files": all_files,
                "common": set(common_features),
                "union": union_features,
            }  # Store group info

        report_rows = generate_pairwise_report(group_info)  # Generate pairwise report rows
        if not report_rows:  # If no report rows were generated
            return False  # Return False indicating failure

        saved_any = False  # Flag to track if any report was saved
        for group_name, info in group_info.items():  # Iterate over each group
            base_dir = (
                os.path.dirname(os.path.abspath(info["files"][0])) if info["files"] else os.getcwd()
            )  # Base dir from first file or current dir
            adjusted_rows = adjust_rows_for_group(report_rows, group_name)  # Adjust rows for this group
            try:  # Try to write the report
                write_report(adjusted_rows, base_dir, output_filename, config=cfg)  # Write the report
                saved_any = True  # Mark that at least one report was saved
            except Exception:  # Fail silently
                pass  # Do nothing on failure

        return saved_any  # Return whether any report was saved
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def to_seconds(obj):
    """
    Converts various time-like objects to seconds.
    
    :param obj: The object to convert (can be int, float, timedelta, datetime, etc.)
    :return: The equivalent time in seconds as a float, or None if conversion fails
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if obj is None:  # None can't be converted
            return None  # Signal failure to convert
        if isinstance(obj, (int, float)):  # Already numeric (seconds or timestamp)
            return float(obj)  # Return as float seconds
        if hasattr(obj, "total_seconds"):  # Timedelta-like objects
            try:  # Attempt to call total_seconds()
                return float(obj.total_seconds())  # Use the total_seconds() method
            except Exception:
                pass  # Fallthrough on error
        if hasattr(obj, "timestamp"):  # Datetime-like objects
            try:  # Attempt to call timestamp()
                return float(obj.timestamp())  # Use timestamp() to get seconds since epoch
            except Exception:
                pass  # Fallthrough on error
        return None  # Couldn't convert
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def calculate_execution_time(start_time, finish_time=None):
    """
    Calculate the execution time and return a human-readable string.

    :param start_time: The start time or duration value (datetime, timedelta, or numeric seconds).
    :param finish_time: Optional finish time; if None, start_time is treated as the total duration.
    :return: Human-readable execution time string formatted as days, hours, minutes, and seconds.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        if finish_time is None:  # Single-argument mode: start_time already represents duration or seconds
            total_seconds = to_seconds(start_time)  # Try to convert provided value to seconds
            if total_seconds is None:  # Conversion failed
                try:  # Attempt numeric coercion
                    total_seconds = float(start_time)  # Attempt numeric coercion
                except Exception:
                    total_seconds = 0.0  # Fallback to zero
        else:  # Two-argument mode: Compute difference finish_time - start_time
            st = to_seconds(start_time)  # Convert start to seconds if possible
            ft = to_seconds(finish_time)  # Convert finish to seconds if possible
            if st is not None and ft is not None:  # Both converted successfully
                total_seconds = ft - st  # Direct numeric subtraction
            else:  # Fallback to other methods
                try:  # Attempt to subtract (works for datetimes/timedeltas)
                    delta = finish_time - start_time  # Try subtracting (works for datetimes/timedeltas)
                    total_seconds = float(delta.total_seconds())  # Get seconds from the resulting timedelta
                except Exception:  # Subtraction failed
                    try:  # Final attempt: Numeric coercion
                        total_seconds = float(finish_time) - float(start_time)  # Final numeric coercion attempt
                    except Exception:  # Numeric coercion failed
                        total_seconds = 0.0  # Fallback to zero on failure

        if total_seconds is None:  # Ensure a numeric value
            total_seconds = 0.0  # Default to zero
        if total_seconds < 0:  # Normalize negative durations
            total_seconds = abs(total_seconds)  # Use absolute value

        days = int(total_seconds // 86400)  # Compute full days
        hours = int((total_seconds % 86400) // 3600)  # Compute remaining hours
        minutes = int((total_seconds % 3600) // 60)  # Compute remaining minutes
        seconds = int(total_seconds % 60)  # Compute remaining seconds

        if days > 0:  # Include days when present
            return f"{days}d {hours}h {minutes}m {seconds}s"  # Return formatted days+hours+minutes+seconds
        if hours > 0:  # Include hours when present
            return f"{hours}h {minutes}m {seconds}s"  # Return formatted hours+minutes+seconds
        if minutes > 0:  # Include minutes when present
            return f"{minutes}m {seconds}s"  # Return formatted minutes+seconds
        return f"{seconds}s"  # Fallback: only seconds
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


def play_sound():
    """
    Play a sound when the program finishes and skip if the operating system is Windows.

    :return: None.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        cfg = {}
        try:
            cfg = get_config() or {}
        except Exception:
            cfg = {}

        sound_cfg = cfg.get("sound", {}) if isinstance(cfg, dict) else {}
        sound_file = sound_cfg.get("file", SOUND_FILE)
        sound_cmds = sound_cfg.get("commands", SOUND_COMMANDS)

        current_os = platform.system()  # Get the current operating system
        if current_os == "Windows":  # If the current operating system is Windows
            return  # Do nothing on Windows by default

        if verify_filepath_exists(sound_file):  # If the sound file exists
            if current_os in sound_cmds:  # Use commands from config or defaults
                os.system(f"{sound_cmds[current_os]} {sound_file}")  # Play the sound
            else:  # Unknown OS mapping
                print(
                    f"{BackgroundColors.RED}The {BackgroundColors.CYAN}{current_os}{BackgroundColors.RED} is not configured in sound.commands. Please add it!{Style.RESET_ALL}"
                )
        else:  # If the sound file does not exist
            print(
                f"{BackgroundColors.RED}Sound file {BackgroundColors.CYAN}{sound_file}{BackgroundColors.RED} not found. Make sure the file exists or set 'sound.file' in config.yaml.{Style.RESET_ALL}"
            )
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


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
