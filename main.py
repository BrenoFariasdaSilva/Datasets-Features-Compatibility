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


def attempt_playwright_export_with_retry(styled_df, output_path, export_kwargs, timeout_ms):
    """
    Attempt to export a styled DataFrame to PNG using Playwright-based dataframe_image with bounded retries.

    :param styled_df: pandas.Styler object to export as a PNG image.
    :param output_path: File system path where the exported PNG will be written.
    :param export_kwargs: Pre-built keyword arguments dict for dfi.export including table_conversion and timeout.
    :param timeout_ms: Timeout in milliseconds used as the fallback kwargs timeout when signature inspection fails.
    :return: None when export succeeded, or the last encountered exception when all attempts failed.
    """

    max_attempts = 5  # Define the maximum number of bounded Playwright export attempts
    e_inner = None  # Track the last exception; None signals success
    for attempt in range(1, max_attempts + 1):  # Retry up to max_attempts times to handle transient failures
        try:  # Try exporting using dataframe_image with Playwright and configured kwargs
            dfi.export(styled_df, output_path, **export_kwargs)  # Export styled DataFrame to PNG using dataframe_image
            print(f"{BackgroundColors.GREEN}[DEBUG] Exported image: {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log successful export for diagnostics
            upscale_image_if_needed(output_path, fallback=False)  # Upscale exported image if below 4k
            print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log absolute save path
            e_inner = None  # Clear last exception on success
            break  # Exit retry loop after successful export
        except TypeError:  # Handle dfi versions that raise TypeError for unexpected kwargs
            try:  # Attempt fallback export using only the minimal supported kwargs
                try:  # Inspect dfi.export signature to determine supported timeout parameter
                    _params_fallback = set(signature(dfi.export).parameters.keys())  # Get set of supported parameter names
                except Exception:  # If signature inspection fails, treat all params as unsupported
                    _params_fallback = set()  # Use empty set as conservative fallback
                kwargs_fb: dict[str, Any] = {"table_conversion": "playwright"}  # Build minimal fallback kwargs
                if "timeout" in _params_fallback:  # Attach timeout only when supported
                    kwargs_fb["timeout"] = timeout_ms  # Attach timeout using the supported parameter name
                dfi.export(styled_df, output_path, **kwargs_fb)  # Retry export with minimal kwargs
                print(f"{BackgroundColors.GREEN}[DEBUG] Exported image (fallback): {BackgroundColors.CYAN}{output_path}{Style.RESET_ALL}")  # Log fallback export success
                upscale_image_if_needed(output_path, fallback=True)  # Upscale exported image after fallback
                print(f"{BackgroundColors.GREEN}[INFO] Table image successfully saved to: {BackgroundColors.CYAN}{os.path.abspath(output_path)}{Style.RESET_ALL}")  # Log save path after fallback
                e_inner = None  # Clear last exception on success
                break  # Exit retry loop after successful fallback export
            except Exception as _inner_e:  # Capture fallback exception for downstream retry/raise handling
                e_inner = _inner_e  # Record inner fallback exception
        except Exception as _e_export:  # Capture general export exceptions for potential retry
            e_inner = _e_export  # Record exception for downstream retry/raise logic

        try:  # Import Playwright TimeoutError for precise timeout detection
            from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError  # Import when available
        except Exception:  # Disable precise detection when import fails
            PlaywrightTimeoutError = None  # Set to None when import is unavailable

        if PlaywrightTimeoutError is not None and isinstance(e_inner, PlaywrightTimeoutError):  # Playwright timeout detected
            if attempt < max_attempts:  # Retry if attempts remain
                time.sleep(0.5)  # Brief pause before retry to allow transient conditions to clear
                print(f"{BackgroundColors.YELLOW}[WARNING] Playwright screenshot timeout, retrying export (attempt {attempt})...{Style.RESET_ALL}")  # Log retry
                continue  # Retry the export
            else:  # Max attempts exhausted; fall through to fallback strategy
                pass  # No-op; allow fallback to execute after loop
        else:  # Non-timeout exception; retry if attempts remain
            if attempt < max_attempts:  # Retry if attempts remain
                time.sleep(0.2)  # Brief pause before next retry
                continue  # Retry the export
            else:  # Attempts exhausted; fall through to fallback strategy
                pass  # No-op; allow fallback to execute after loop

    return e_inner  # Return None on success or the last exception on failure


def load_tableau_image_config():
    """
    Load the configuration file, resolve the table image timeout, and build the base dataframe_image export kwargs.

    :return: Tuple of (timeout_ms, export_kwargs) where timeout_ms is the configured timeout in milliseconds and export_kwargs is a dict pre-populated with the Playwright conversion option and timeout parameters.
    """

    cfg = load_config_file()  # Load configuration from config.yaml if present in the workspace
    timeout_ms = int((cfg or {}).get("dataset_descriptor", {}).get("table_image_timeout_ms", 30000))  # Determine timeout in milliseconds using config value with hardcoded fallback
    src = "config" if (cfg or {}).get("dataset_descriptor", {}).get("table_image_timeout_ms") is not None else "default"  # Identify whether the timeout came from config or the default value
    print(f"{BackgroundColors.GREEN}[CONFIG] table_image_timeout_ms = {BackgroundColors.CYAN}{timeout_ms}{Style.RESET_ALL} (source: {src})")  # Log the active timeout value and its source with colored terminal output
    export_kwargs: dict[str, Any] = {"table_conversion": "playwright"}  # Build base export kwargs with Playwright as the conversion engine
    export_kwargs["timeout"] = timeout_ms  # Inject the configured timeout so Playwright receives the correct value
    try:  # Inspect dfi.export signature to attach the matching screenshot timeout parameter name
        params = set(signature(dfi.export).parameters.keys())  # Retrieve the set of parameter names supported by dfi.export
        for _pname in ("screenshot_timeout", "timeout", "playwright_timeout", "playwright_screenshot_timeout"):  # Iterate candidate timeout parameter names from various dfi versions
            if _pname in params:  # Verify whether this candidate is present in the detected parameter set
                export_kwargs[_pname] = timeout_ms  # Attach the timeout using the first matching parameter name
                break  # Stop after the first supported parameter to avoid conflicting kwargs
    except Exception:  # Ignore signature inspection failures to preserve original behavior
        pass  # Continue without explicit screenshot timeout when inspection is unavailable
    return timeout_ms, export_kwargs  # Return the resolved timeout and fully built export kwargs dict


def export_dataframe_image(styled_df, output_path):
    """
    Export a pandas.Styler to a PNG image using dataframe_image.

    :param styled_df: pandas.Styler object to export
    :param output_path: Path to write PNG image
    :return: None
    """

    try:  # Wrap to ensure exceptions are handled and module logging conventions are preserved
        timeout_ms, export_kwargs = load_tableau_image_config()  # Load config, resolve the table image timeout, and build Playwright export kwargs

        e_inner = attempt_playwright_export_with_retry(styled_df, output_path, export_kwargs, timeout_ms)  # Attempt export with bounded Playwright retries and return last exception or None on success

        if e_inner is not None:  # If last Playwright/dfi attempt failed and no success occurred
            e_inner = attempt_chrome_export_fallback(styled_df, output_path, e_inner, export_kwargs, timeout_ms)  # Try Chrome as first deterministic fallback and update e_inner

        if e_inner is not None:  # If both Playwright and chrome fallbacks failed, attempt matplotlib rendering as last resort
            e_inner = attempt_matplotlib_export_fallback(styled_df, output_path, e_inner)  # Try matplotlib as final fallback and update e_inner

        if e_inner is not None:  # If all methods failed, re-raise the last encountered exception to be handled by outer block
            raise e_inner  # Re-raise last exception to preserve original outer logging and telemetry behavior
    except Exception as e:  # If export fails, log warning and continue without crashing
        try:  # Try to import Playwright-specific TimeoutError for precise detection
            from playwright._impl._errors import TimeoutError as PlaywrightTimeoutError  # Optional import of Playwright TimeoutError for specific handling
        except Exception:  # If import fails, ensure variable is defined for downstream verification logic
            PlaywrightTimeoutError = None  # Set to None when Playwright TimeoutError cannot be imported
        if PlaywrightTimeoutError is not None and isinstance(e, PlaywrightTimeoutError):  # Verify if exception is Playwright TimeoutError
            print(f"{BackgroundColors.YELLOW}[WARNING] Playwright screenshot timeout while exporting {BackgroundColors.CYAN}{output_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn when Playwright timeout occurs with colored output
        else:  # General failure when not a Playwright TimeoutError
            print(f"{BackgroundColors.YELLOW}[WARNING] Failed to export image {BackgroundColors.CYAN}{output_path}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn for general export failures with colored output
        return  # Return gracefully to avoid terminating the program
    finally:  # Ensure multiprocessing and large object cleanup regardless of export outcome
        try:  # Attempt explicit multiprocessing resource finalization to avoid leaked semaphores
            finalize_multiprocessing_resources()  # Finalize active child processes and resource tracker state
        except Exception:  # Ignore cleanup failures to preserve non-fatal export semantics
            pass  # Continue gracefully when finalization fails
        try:  # Attempt to release styled object reference as soon as export flow ends
            del styled_df  # Delete styled DataFrame reference to reduce retained memory
        except Exception:  # Ignore delete failures to preserve behavior
            pass  # Continue gracefully when reference deletion fails
        gc.collect()  # Trigger garbage collection after export cleanup


def generate_table_image_from_dataframe(df, output_path, config: dict | None = None):
    """
    Generate a zebra-striped PNG table image from a DataFrame and save to output_path.

    :param df: pandas.DataFrame to render.
    :param output_path: Path for output PNG image.
    :param config: Optional configuration dictionary (reserved for future use).
    :return: None.
    """

    try:  # Wrap to preserve module's error handling conventions
        styled = apply_zebra_style(df)  # Create a styled DataFrame with zebra striping
        export_dataframe_image(styled, output_path)  # Export the styled DataFrame to PNG
    except Exception:  # Do not swallow exceptions here per spec
        raise  # Re-raise any exception to caller


def generate_csv_and_image(df, csv_path, config: dict | None = None):
    """
    Save a DataFrame to CSV and generate a corresponding PNG table image next to it.

    :param df: pandas.DataFrame to save and render.
    :param csv_path: Full path for CSV output.
    :param config: Optional configuration dictionary for resolving image format.
    :return: Tuple (csv_path, image_path).
    """

    try:  # Wrap to preserve module's error handling conventions
        if not isinstance(csv_path, str) or not csv_path:  # Verify csv_path is a non-empty string
            raise ValueError("csv_path must be a non-empty string")  # Raise when csv_path is missing or invalid
        df.to_csv(csv_path, index=False)  # Persist DataFrame to CSV without index
        img_ext = (config or {}).get("dataset_descriptor", {}).get("table_image_format", "png")  # Resolve table image format from config with png fallback
        image_path = os.path.splitext(csv_path)[0] + f".{img_ext}"  # Build image path using configured format
        if len(df) <= 100:  # Generate image only when DataFrame size is within the safe row limit
            try:  # Guard image rendering to preserve the already-written CSV on PNG export failure
                generate_table_image_from_dataframe(df, image_path, config=config)  # Generate image from DataFrame
            except Exception as e:  # Contain PNG export failure locally to avoid aborting the pipeline
                print(f"{BackgroundColors.YELLOW}[WARNING] Failed to generate table image for {BackgroundColors.CYAN}{os.path.basename(csv_path)}{BackgroundColors.YELLOW}: {e}{Style.RESET_ALL}")  # Warn and continue when PNG rendering fails
        return csv_path, image_path  # Return both paths for caller use
    except Exception:
        raise  # Re-raise to preserve original failure semantics


def finalize_and_write_report(report_rows, preprocessing_metrics, base_dir, output_filename, config):
    """
    Number report rows, write the report CSV, generate the preprocessing summary, and return the success flag.

    :param report_rows: List of per-file info dicts accumulated during the processing loop.
    :param preprocessing_metrics: List of per-file preprocessing metric dicts for summary generation.
    :param base_dir: Absolute base directory used as the output root for report and summary files.
    :param output_filename: Resolved output filename string ending with ".csv".
    :param config: Optional configuration dictionary forwarded to write_report and save_preprocessing_summary_csv.
    :return: True when the report was written successfully, False when no report rows were available.
    """

    if not report_rows:  # Return False immediately when the processing loop produced no data rows
        return False  # Signal failure to the caller when no rows were collected
    for i, row in enumerate(report_rows, start=1):  # Assign sequential row numbers starting at 1
        row["#"] = i  # Embed the counter directly into each row dict before writing
    write_report(report_rows, base_dir, output_filename, config=config)  # Persist all numbered rows as the main report CSV
    try:  # Generate the preprocessing summary separately to avoid aborting the main report on failure
        if preprocessing_metrics:  # Only generate a summary when per-file metrics were successfully collected
            pre_df = build_preprocessing_summary_dataframe(preprocessing_metrics)  # Build a DataFrame from the accumulated metrics list
            out_path = save_preprocessing_summary_csv(pre_df, base_dir, config=config)  # Save the summary CSV to the results directory
            print(f"{BackgroundColors.GREEN}Saved preprocessing summary to {BackgroundColors.CYAN}{out_path}{Style.RESET_ALL}")  # Inform the user of the saved summary path
            if os.environ.get("DD_DESCRIPTOR_VERBOSE", "False").lower() in ("1", "true", "yes"):  # Print table only in verbose mode
                print_preprocessing_summary_table(pre_df)  # Print the summary table to the terminal when verbose output is enabled
    except Exception as _ps:  # Warn and continue when summary generation fails to preserve the main report
        print(f"{BackgroundColors.YELLOW}Warning: failed to generate preprocessing summary: {_ps}{Style.RESET_ALL}")  # Warn without aborting
    return True  # Return True to signal that the main report was written successfully


def resolve_output_filename(output_filename, cfg):
    """
    Resolve the output filename for the dataset report CSV, applying config defaults and ensuring a .csv extension.

    :param output_filename: Caller-provided filename string, or None to use a config-derived default.
    :param cfg: Configuration dictionary used to read the csv_output_suffix fallback value.
    :return: Resolved output filename string guaranteed to end with ".csv".
    """

    if output_filename is None:  # Use config-based suffix when no filename was provided by the caller
        output_filename = cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "description")  # Read configured suffix with hardcoded fallback
    if not isinstance(output_filename, str):  # Convert non-string filename to string using config suffix as fallback
        output_filename = str(output_filename or cfg.get("dataset_descriptor", {}).get("csv_output_suffix", "_description"))  # Stringify with config fallback when value is falsy
    if not output_filename.lower().endswith(".csv"):  # Append .csv extension when absent
        output_filename = f"{output_filename}.csv"  # Ensure the filename always has the .csv extension
    return output_filename  # Return the fully resolved output filename


def collect_report_input_files(input_path, file_extension, config):
    """
    Determine the matching files and base directory from the provided input path.

    :param input_path: Directory or single file path to scan for matching dataset files.
    :param file_extension: File extension used to filter files when scanning a directory.
    :param config: Optional configuration dictionary passed through to collect_matching_files.
    :return: Tuple of (sorted_matching_files, base_dir) where sorted_matching_files is a list of absolute file paths and base_dir is the absolute base directory used for relative path computations.
    """

    if os.path.isdir(input_path):  # Scan the directory for all matching files
        print(
            f"{BackgroundColors.GREEN}Scanning directory {BackgroundColors.CYAN}{input_path}{BackgroundColors.GREEN} for {BackgroundColors.CYAN}{file_extension}{BackgroundColors.GREEN} files...{Style.RESET_ALL}"
        )  # Announce directory scan start
        sorted_matching_files = collect_matching_files(input_path, file_extension, config=config)  # Collect all matching files from the directory tree
        base_dir = os.path.abspath(input_path)  # Use the directory itself as the base for relative paths
    elif os.path.isfile(input_path) and input_path.endswith(file_extension):  # Single file provided
        print(
            f"{BackgroundColors.GREEN}Processing single file...{Style.RESET_ALL}"
        )  # Announce single file processing
        sorted_matching_files = [input_path]  # Wrap the single file in a list for uniform processing
        base_dir = os.path.dirname(os.path.abspath(input_path))  # Use the file's parent directory as base
    else:  # Input is neither a directory nor a valid matching file
        print(
            f"{BackgroundColors.RED}Input path is neither a directory nor a valid {file_extension} file: {input_path}{Style.RESET_ALL}"
        )  # Report the invalid input path
        sorted_matching_files = []  # No files to process when input is invalid
        base_dir = os.path.abspath(input_path)  # Preserve input path as base for any downstream error messages
    return sorted_matching_files, base_dir  # Return the collected files and resolved base directory


def enrich_file_info_with_metadata(info, filepath, base_dir, headers_map, common_features, headers_match_all, cfg, low_memory, df_current):
    """
    Populate a file info dictionary with relative path, header uniformity, common/extra feature lists, and t-SNE plot path.

    :param info: Mutable dictionary of dataset metadata fields populated in place by this function.
    :param filepath: Absolute path of the dataset file being processed.
    :param base_dir: Absolute base directory used to compute the relative path for the Dataset Name field.
    :param headers_map: Dictionary mapping file paths to their header lists for common/extra feature computation.
    :param common_features: Set of feature names present in every discovered file used for common/extra classification.
    :param headers_match_all: Boolean flag indicating whether all files share identical header sets.
    :param cfg: Configuration dictionary used to look up the t-SNE output subdirectory key.
    :param low_memory: Boolean flag passed to the t-SNE generator to control memory usage during plot generation.
    :param df_current: Already-loaded pandas DataFrame for the current file passed to the t-SNE generator to avoid a second disk read.
    :return: None (modifies info in place with Dataset Name, Headers Match All Files, Common Features, Extra Features, and t-SNE Plot fields).
    """

    relative_path = os.path.relpath(filepath, base_dir)  # Get path relative to base_dir
    info["Dataset Name"] = relative_path.replace(
        "\\", "/"
    )  # Use relative path for Dataset Name and normalize slashes

    common_list, extras = get_file_common_and_extras(
        headers_map, filepath, common_features
    )  # Get common and extra features for this file

    info["Headers Match All Files"] = (
        "Yes" if headers_match_all else "No"
    )  # Indicate if headers match all files
    info["Common Features (in all files)"] = (
        ", ".join(common_list) if common_list else "None"
    )  # Join common features into a string
    info["Extra Features (not in all files)"] = (
        ", ".join(extras) if extras else "None"
    )  # Join extra features into a string

    tsne_out_subdir = cfg.get("paths", {}).get("data_separability_subdir", "Data_Separability")  # Read t-SNE output subdirectory name from configuration with default fallback
    tsne_file = generate_tsne_plot(
        filepath,
        df=df_current,
        low_memory=low_memory,
        sample_size=2000,
        output_dir=os.path.join(os.path.dirname(os.path.abspath(filepath)), tsne_out_subdir),
        config=cfg,
    )  # Generate t-SNE plot using the already-loaded DataFrame to avoid rereading from disk
    info["t-SNE Plot"] = tsne_file if tsne_file else "None"  # Add t-SNE plot filename or "None"


def append_preprocessing_metrics_safe(filepath, info, preprocessing_metrics, file_basename):
    """
    Collect preprocessing metrics for a processed file and append them to the metrics list, WARNING on failure.

    :param filepath: Absolute path of the dataset file used as identifier in the metrics row.
    :param info: Dataset metadata dictionary providing original and post-preprocessing row/feature counts.
    :param preprocessing_metrics: Mutable list to which the collected metrics row dictionary is appended.
    :param file_basename: Relative file path string used in the failure warning message for user context.
    :return: None (appends to preprocessing_metrics in place or prints a warning on failure).
    """

    try:  # Collect preprocessing metrics for this file when available
        metrics_row = collect_preprocessing_metrics(
            filepath,  # File path being processed
            info.get("original_num_rows", 0),  # Original rows captured earlier
            info.get("rows_after_preprocessing", 0),  # Rows after preprocessing captured earlier
            info.get("original_num_features", 0),  # Original features captured earlier
            info.get("features_after_preprocessing", 0),  # Features after preprocessing captured earlier
            info.get("rows_after_nan_inf_removal", 0),  # Rows after NaN/infinite removal step
            info.get("removed_rows_nan_inf", 0),  # Rows removed by NaN/infinite filtering step
            info.get("removed_rows_nan_inf_proportion", 0.0),  # Proportion of rows removed by NaN/infinite filtering step
            info.get("features_after_zero_variance_removal", 0),  # Features after zero-variance numerical feature removal step
            info.get("removed_zero_variance_features", 0),  # Zero-variance numerical features removed in preprocessing
            info.get("removed_zero_variance_features_proportion", 0.0),  # Proportion of zero-variance numerical features removed
            info.get("dropped_non_informative_features", 0),  # Non-informative identifier/metadata features removed in this module
            info.get("dropped_non_informative_features_proportion", 0.0),  # Proportion of non-informative identifier/metadata features removed
            info.get("features_transformed_for_experiment", 0),  # Features transformed for dtype enforcement and categorical encoding per experiment
            info.get("features_transformed_for_experiment_proportion", 0.0),  # Proportion of transformed features for dtype enforcement and categorical encoding per experiment
            info.get("features_cast_to_float64_int64", 0),  # Numeric features requiring cast to float64/int64
            info.get("features_encoded_categorical", 0),  # Categorical features requiring ordinal or one-hot encoding
            info.get("preprocessing_metrics", None),  # Structured per-step preprocessing metrics for isolated CSV mapping
        )  # Create metrics row dict
        preprocessing_metrics.append(metrics_row)  # Append metrics row to list for this directory
    except Exception as _pm:  # If metrics collection fails
        print(f"{BackgroundColors.YELLOW}Warning: failed to collect preprocessing metrics for {file_basename}: {_pm}{Style.RESET_ALL}")  # Warn without breaking the progress bar


def generate_dataset_report(input_path, file_extension=".csv", low_memory=None, output_filename: str | None = None, config: dict | None = None):
    """
    Generate a CSV report for the specified input path.
    The Dataset Name column will include subdirectories if present.

    :param input_path: Directory or file path containing the dataset.
    :param file_extension: File extension to filter (default: .csv).
    :param low_memory: Whether to use low memory mode when loading CSVs (default: True).
    :param output_filename: Name of the CSV file to save the report.
    :param config: Optional configuration dictionary for resolving paths and settings.
    :return: True if the report was generated successfully, False otherwise.
    """

    try:  # Wrap full function logic to ensure production-safe monitoring
        report_rows = []  # List to store report rows
        sorted_matching_files = []  # List to store matching files
        preprocessing_metrics = []  # List to collect per-file preprocessing metric dicts

        sorted_matching_files, base_dir = collect_report_input_files(input_path, file_extension, config)  # Collect matching files and resolve base directory from the provided input path

        cfg = config or get_default_config()

        if not sorted_matching_files:  # If no matching files were found
            print(f"{BackgroundColors.RED}No matching {file_extension} files found in: {input_path}{Style.RESET_ALL}")
            return False  # Exit the function

        output_filename = resolve_output_filename(output_filename, cfg)  # Resolve the output filename, applying config defaults and ensuring a .csv extension

        headers_map = build_headers_map(sorted_matching_files, low_memory=low_memory)  # Build headers map using lightweight header-only reads to avoid loading all datasets into memory simultaneously
        common_features, headers_match_all = compute_common_features(headers_map)  # Compute shared features and header uniformity flag from the headers-only map

        progress = tqdm(
            sorted_matching_files,  # Iterate over sorted matching files list
            desc=f"{BackgroundColors.GREEN}Processing files{Style.RESET_ALL}",  # Description text remains green and reset styles
            unit="file",  # Use file as unit for progress
            ncols=100,  # Fixed progress bar width in characters
            colour="cyan",  # Set progress bar visualization color to cyan
        )  # Create a single in-place progress bar instance
        for idx, filepath in enumerate(progress, 1):  # Process each matching file
            file_basename = os.path.relpath(filepath, base_dir).replace("\\", "/")  # Get the file path relative to base_dir and normalize slashes
            colored_desc = f"{BackgroundColors.GREEN}Processing {BackgroundColors.CYAN}{file_basename}{Style.RESET_ALL}"  # Compose colored description using BackgroundColors while keeping length bounded
            progress.set_description(colored_desc)  # Update progress bar description with colored, truncated filename for inline display

            df_current = load_dataset(filepath, low_memory)  # Load one dataset at a time to minimize peak RAM usage
            if df_current is None:  # Verify that the dataset was loaded successfully
                print(f"{BackgroundColors.YELLOW}Warning: failed to load {filepath}; skipping.{Style.RESET_ALL}")  # Warn about the skipped file without breaking the progress bar
                continue  # Skip to the next file without accumulating a None entry

            info = get_dataset_file_info(filepath, df=df_current, low_memory=low_memory)  # Extract metadata using the already-loaded DataFrame to avoid a second full read
            if info:  # If info was successfully retrieved
                enrich_file_info_with_metadata(info, filepath, base_dir, headers_map, common_features, headers_match_all, cfg, low_memory, df_current)  # Populate Dataset Name, Headers Match, Common/Extra Features, and t-SNE Plot fields in place

                report_rows.append(info)  # Add the info to the report rows

                append_preprocessing_metrics_safe(filepath, info, preprocessing_metrics, file_basename)  # Collect and append preprocessing metrics with error-safe handling
                del info  # Release info reference after appending to report structures to reduce retention

            try:  # Attempt to release dataset memory to minimize peak RAM consumption
                del df_current  # Delete the current dataset reference to allow garbage collection
            except Exception:  # Ignore exceptions during cleanup to prevent masking processing errors
                pass  # Continue without cleanup on delete failure
            gc.collect()  # Force garbage collection to reclaim memory released by deleting the dataset

        return finalize_and_write_report(report_rows, preprocessing_metrics, base_dir, output_filename, config)  # Number rows, write report, generate preprocessing summary, and return success flag
    except Exception as e:  # Catch any exception to ensure logging
        print(str(e))  # Print error to terminal for server logs
        raise  # Re-raise to preserve original failure semantics


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
