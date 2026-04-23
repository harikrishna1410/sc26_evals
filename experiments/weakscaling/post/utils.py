import csv
import glob
import os
import re
from datetime import datetime

# ============================================================================
# TIMESTAMP UTILITIES
# ============================================================================


def get_log_timestamp(log_path):
    """
    Get the timestamp of the last line in a log file.

    Args:
        log_path: Path to the log file

    Returns:
        datetime: Timestamp of the last line, or None if cannot parse
    """
    try:
        with open(log_path, "r") as f:
            lines = f.readlines()
            if not lines:
                return None

            # Read last non-empty line
            for line in reversed(lines):
                if line.strip():
                    # Try to extract timestamp from beginning of line
                    timestamp_str = line.split(" - ")[0].strip()
                    try:
                        return datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    except ValueError:
                        continue
            return None
    except (FileNotFoundError, IOError):
        return None


def get_latest_log_timestamp(
    system_type, base_dir, nn, nworkers, ntasks_per_worker, sleep_time
):
    """
    Get the latest timestamp from log files for a given configuration.

    Args:
        system_type: Type of system ('dask', 'parsl', 'flux', 'el')
        base_dir: Base directory for the system
        nn: Number of nodes
        nworkers: Number of workers per node
        ntasks_per_worker: Number of tasks per worker
        sleep_time: Task sleep time in seconds

    Returns:
        datetime: Latest timestamp across all relevant log files, or None
    """
    timestamps = []

    if system_type in ["dask", "parsl"]:
        for i in range(1, 4):
            log_file = f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}s_{i}/main.log"
            ts = get_log_timestamp(log_file)
            if ts:
                timestamps.append(ts)

    elif system_type == "flux":
        for i in range(1, 4):
            log_dir = f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}s_{i}"
            # Find all *_main.log files
            log_files = glob.glob(os.path.join(log_dir, "*_main.log"))
            for log_file in log_files:
                ts = get_log_timestamp(log_file)
                if ts:
                    timestamps.append(ts)

    elif system_type == "el":
        for i in range(1, 4):
            log_dir = f"{base_dir}/{nn}/all_logs/logs_{nworkers}_{ntasks_per_worker}_{sleep_time}s_{i}_async"
            # Find all *.w*.log files
            log_files = glob.glob(os.path.join(log_dir, "*.w*.log"))
            for log_file in log_files:
                ts = get_log_timestamp(log_file)
                if ts:
                    timestamps.append(ts)

    return max(timestamps) if timestamps else None


# ============================================================================
# THROUGHPUT CALCULATION
# ============================================================================


def get_effective_task_throughput(log_file_path):
    """
    Calculate effective task throughput from a log file.

    Throughput is calculated as:
        total_tasks / time_between_running_and_finished

    Args:
        log_file_path: Path to the log file

    Returns:
        dict: Dictionary containing:
            - ntasks: Number of tasks
            - start_time: Timestamp when tasks started running
            - end_time: Timestamp when tasks finished
            - duration: Time difference in seconds
            - throughput: Tasks per second
    """
    with open(log_file_path, "r") as f:
        lines = f.readlines()

    start_time = None
    end_time = None
    ntasks = None

    # Regex patterns
    running_pattern = re.compile(r"Running (\d+) ([a-zA-Z]+) tasks on (\d+) nodes")
    finished_pattern = re.compile(r"All ([a-zA-Z]+) tasks finished")

    for line in lines:
        # Look for "Running X tasks" line (any type of tasks)
        match = running_pattern.search(line)
        if match:
            # Extract timestamp
            timestamp_str = line.split(" - ")[0]
            start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
            # Extract number of tasks
            ntasks = int(match.group(1))

        # Look for "All tasks finished" line (any type of tasks)
        if finished_pattern.search(line):
            # Extract timestamp
            timestamp_str = line.split(" - ")[0]
            end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

    if start_time is None or end_time is None or ntasks is None:
        raise ValueError("Could not find required lines in log file")

    # Calculate duration in seconds
    duration = (end_time - start_time).total_seconds()

    # Calculate throughput
    throughput = ntasks / duration if duration > 0 else 0

    return {
        "ntasks": ntasks,
        "start_time": start_time,
        "end_time": end_time,
        "duration": duration,
        "throughput": throughput,
    }


def get_effective_task_throughput_dask(log_file_path):
    """
    Calculate effective task throughput from a Dask log file.

    Wrapper for get_effective_task_throughput.

    Args:
        log_file_path: Path to the log file

    Returns:
        dict: Dictionary containing throughput metrics
    """
    return get_effective_task_throughput(log_file_path)


def get_effective_task_throughput_parsl(log_file_path):
    """
    Calculate effective task throughput from a Parsl log file.

    Wrapper for get_effective_task_throughput.

    Args:
        log_file_path: Path to the log file

    Returns:
        dict: Dictionary containing throughput metrics
    """
    return get_effective_task_throughput(log_file_path)


def get_effective_task_throughput_flux(log_dir):
    """
    Calculate effective task throughput from Flux log files.

    Scans a directory for all *_main.log files, extracts start/end times
    and number of tasks from each, then computes aggregate throughput.

    Throughput is calculated as:
        total_tasks / (max_end_time - min_start_time)

    Args:
        log_dir: Directory containing *_main.log files

    Returns:
        dict: Dictionary containing:
            - ntasks: Total number of tasks across all files
            - start_time: Earliest start time across all files
            - end_time: Latest end time across all files
            - duration: Time difference in seconds
            - throughput: Tasks per second
            - num_files: Number of log files processed
    """
    # Find all *_main.log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*_main.log"))

    if not log_files:
        raise ValueError(f"No *_main.log files found in {log_dir}")

    all_start_times = []
    all_end_times = []
    all_ntasks = []

    # Regex patterns for Flux logs
    submitting_pattern = re.compile(
        r"Submitting (\d+) tasks to each of (\d+) flux URIs"
    )
    finished_pattern = re.compile(r"All (\d+) tasks finished")

    for log_file in log_files:
        with open(log_file, "r") as f:
            lines = f.readlines()

        start_time = None
        end_time = None
        ntasks = None

        for line in lines:
            # Look for "Submitting X tasks to each of Y flux URIs" line
            match = submitting_pattern.search(line)
            if match:
                # Extract timestamp
                timestamp_str = line.split(" - ")[0]
                start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                # Extract number of tasks (tasks per URI * number of URIs)
                tasks_per_uri = int(match.group(1))
                num_uris = int(match.group(2))
                ntasks = tasks_per_uri * num_uris

            # Look for "All X tasks finished" line
            match = finished_pattern.search(line)
            if match:
                # Extract timestamp
                timestamp_str = line.split(" - ")[0]
                end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

        if start_time and end_time and ntasks:
            all_start_times.append(start_time)
            all_end_times.append(end_time)
            all_ntasks.append(ntasks)

    if not all_start_times:
        raise ValueError("Could not find required lines in any log files")

    # Aggregate across all files
    min_start_time = min(all_start_times)
    max_end_time = max(all_end_times)
    total_tasks = sum(all_ntasks)

    # Calculate duration in seconds
    duration = (max_end_time - min_start_time).total_seconds()

    # Calculate throughput
    throughput = total_tasks / duration if duration > 0 else 0

    return {
        "ntasks": total_tasks,
        "start_time": min_start_time,
        "end_time": max_end_time,
        "duration": duration,
        "throughput": throughput,
        "num_files": len(all_start_times),
    }


def get_effective_task_throughput_el(log_dir):
    """
    Calculate effective task throughput from Ensemble Launcher log files.

    Scans a directory for all *.w*.log files (worker logs), extracts start/end times
    from each, and computes aggregate throughput.

    The directory name should follow the pattern:
        logs_{nconcurrent_workers}_{ntasks_per_worker}_{idealruntime}_{iteration}_{mode}

    Total tasks = nconcurrent_workers * ntasks_per_worker * number_of_worker_log_files

    Throughput is calculated as:
        total_tasks / (max_end_time - min_start_time)

    Args:
        log_dir: Directory containing *.w*.log files with directory name pattern

    Returns:
        dict: Dictionary containing:
            - ntasks: Total number of tasks across all files
            - start_time: Earliest start time across all files
            - end_time: Latest end time across all files
            - duration: Time difference in seconds
            - throughput: Tasks per second
            - num_files: Number of log files processed
            - nconcurrent_workers: Number of concurrent workers (from dir name)
            - ntasks_per_worker: Number of tasks per worker (from dir name)
    """
    # Find all *.w*.log files in the directory
    log_files = glob.glob(os.path.join(log_dir, "*.w*.log"))

    if not log_files:
        raise ValueError(f"No *.w*.log files found in {log_dir}")

    # Parse directory name to get parameters
    # Expected format: logs_{nconcurrent_workers}_{ntasks_per_worker}_{idealruntime}_{iteration}_{mode}
    dir_name = os.path.basename(log_dir)
    parts = dir_name.split("_")

    if len(parts) < 3:
        raise ValueError(
            f"Directory name '{dir_name}' does not follow expected pattern"
        )

    try:
        # parts[0] = "logs"
        # parts[1] = nconcurrent_workers
        # parts[2] = ntasks_per_worker
        nconcurrent_workers = int(parts[1])
        ntasks_per_worker = int(parts[2])
    except (ValueError, IndexError):
        raise ValueError(
            f"Could not parse nconcurrent_workers and ntasks_per_worker from '{dir_name}'"
        )

    all_start_times = []
    all_end_times = []

    for log_file in log_files:
        with open(log_file, "r") as f:
            lines = f.readlines()

        start_time = None
        end_time = None

        for line in lines:
            # Look for "Received task update from parent" line
            if "Received task update from parent" in line:
                # Extract timestamp
                timestamp_str = line.split(" - ")[0]
                start_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

            # Look for "All tasks completed" line
            if "All tasks completed" in line:
                # Extract timestamp
                timestamp_str = line.split(" - ")[0]
                end_time = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

        if start_time and end_time:
            all_start_times.append(start_time)
            all_end_times.append(end_time)

    if not all_start_times:
        raise ValueError("Could not find required lines in any log files")

    # Calculate total tasks
    num_worker_files = len(all_start_times)
    total_tasks = nconcurrent_workers * ntasks_per_worker * num_worker_files

    # Aggregate across all files
    min_start_time = min(all_start_times)
    max_end_time = max(all_end_times)

    # Calculate duration in seconds
    duration = (max_end_time - min_start_time).total_seconds()

    # Calculate throughput
    throughput = total_tasks / duration if duration > 0 else 0

    return {
        "ntasks": total_tasks,
        "start_time": min_start_time,
        "end_time": max_end_time,
        "duration": duration,
        "throughput": throughput,
        "num_files": num_worker_files,
        "nconcurrent_workers": nconcurrent_workers,
        "ntasks_per_worker": ntasks_per_worker,
    }


# ============================================================================
# TASK DATA READING UTILITIES
# ============================================================================

import json
import subprocess

import pandas as pd


def build_merged_profile(profiles_dir, output_file):
    """
    Build merged_profile.json from profile traces using ensemble_launcher.

    Parameters:
    -----------
    profiles_dir : str
        Directory containing individual profile JSON files
    output_file : str
        Path to output merged profile JSON file
    """
    if not os.path.exists(profiles_dir):
        print(f"Warning: Profiles directory not found: {profiles_dir}")
        return False

    cmd = [
        "python3",
        "-m",
        "ensemble_launcher.profiling.merge_traces",
        "--input-dir",
        profiles_dir,
        "--output",
        output_file,
        "--no-timestamp-merge",
    ]

    try:
        print(f"Building merged profile: {output_file}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Successfully created: {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error building merged profile: {e}")
        print(f"stderr: {e.stderr}")
        return False


def merge_flux_timelines(log_dir, output_file, offsets=None):
    """
    Merge multiple Flux *_timeline.csv files into a single merged_timeline.csv.
    Adjusts task_id by adding offset based on previous files' task counts.
    Optionally applies time offsets to completion_time.

    Parameters:
    -----------
    log_dir : str
        Directory containing *_timeline.csv files
    output_file : str
        Path to output merged_timeline.csv file
    offsets : dict, optional
        Dictionary mapping timeline base names to time offsets in seconds
    """
    pattern = os.path.join(log_dir, "*_timeline.csv")
    timeline_files = sorted(glob.glob(pattern))

    if not timeline_files:
        print(f"Warning: No timeline CSV files found in {log_dir}")
        return False

    all_dfs = []
    task_offset = 0

    for csv_file in timeline_files:
        try:
            df = pd.read_csv(csv_file)

            # Adjust task_id by adding offset from previous files
            if "task_id" in df.columns:
                df["task_id"] = df["task_id"] + task_offset
                task_offset += len(df)

            # Apply time offset if provided
            if offsets is not None and "completion_time" in df.columns:
                base_name = os.path.basename(csv_file).replace("_timeline.csv", "")
                if base_name in offsets:
                    time_offset = offsets[base_name]
                    # Make times relative to start of this file, then add offset
                    start_time = df["completion_time"].min()
                    df["completion_time"] = (
                        df["completion_time"] - start_time + time_offset
                    )

            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Error reading {csv_file}: {e}")
            continue

    if not all_dfs:
        print(f"Warning: Could not read any timeline files from {log_dir}")
        return False

    # Merge all dataframes and sort by completion_time
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.sort_values("completion_time")

    # Reset task_number to be sequential
    merged_df["task_number"] = range(1, len(merged_df) + 1)

    # Save merged timeline
    merged_df.to_csv(output_file, index=False)
    print(f"Merged {len(timeline_files)} timeline files into: {output_file}")
    return True


def read_perfetto_profile(json_file, worker_offsets=None):
    """
    Read task completion data from a Perfetto/Chrome trace format JSON file.
    Extracts task_execution end events relative to launch_children start.

    Parameters:
    -----------
    json_file : str
        Path to the merged_profile.json file
    worker_offsets : dict, optional
        Dictionary mapping worker_id -> time offset in seconds.
        If provided, adjusts event timestamps based on worker offsets.

    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: task_id, completion_time (relative, in seconds)
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])

    # Find the launch_children event to use as time reference
    launch_time = None
    for event in trace_events:
        if (
            event.get("cat") == "launch_children"
            or "launch" in event.get("name", "").lower()
        ):
            # Use the timestamp of the first launch-related event
            launch_time = 0
            break

    # If no launch event found, use the earliest timestamp
    if launch_time is None:
        launch_time = min((e.get("ts", float("inf")) for e in trace_events), default=0)

    # Extract task completion events (phase "e" with category "task_execution")
    completions = []
    for event in trace_events:
        if event.get("cat") == "task_execution" and event.get("ph") == "e":
            task_name = event.get("name", "")
            ts = event.get("ts", 0)

            # Convert to seconds relative to launch time
            completion_time = (ts - launch_time) / 1e6

            # Apply worker offset if available
            if worker_offsets is not None:
                # Extract node_id from event args
                args = event.get("args", {})
                node_id = args.get("node_id")
                if node_id and node_id in worker_offsets:
                    # Add offset to align worker timeline with master timeline
                    completion_time += worker_offsets[node_id]

            completions.append(
                {"task_id": task_name, "completion_time": completion_time}
            )

    if not completions:
        print(f"Warning: No task completion events found in {json_file}")
        return pd.DataFrame(columns=["task_id", "completion_time"])

    df = pd.DataFrame(completions)
    df = df.sort_values("completion_time")
    return df


def get_task_latencies_from_csv(csv_file):
    """
    Extract task latencies from CSV file using elapsed_time column.

    Parameters:
    -----------
    csv_file : str
        Path to the timeline CSV file

    Returns:
    --------
    list
        List of task latencies in seconds
    """
    df = pd.read_csv(csv_file)

    # For CSV files, use elapsed_time if available
    if "elapsed_time" in df.columns:
        latencies = df["elapsed_time"].tolist()
    elif "completion_time" in df.columns and "start_time" in df.columns:
        latencies = (df["completion_time"] - df["start_time"]).tolist()
    else:
        print(f"Warning: Cannot find latency columns in {csv_file}")
        return []

    return latencies


def get_task_latencies_from_json(json_file):
    """
    Extract task latencies from Perfetto JSON file.
    Matches task_execution events with ph='b' and ph='e' to compute duration.

    Parameters:
    -----------
    json_file : str
        Path to the merged_profile.json file

    Returns:
    --------
    list
        List of task latencies in seconds
    """
    with open(json_file, "r") as f:
        data = json.load(f)

    trace_events = data.get("traceEvents", [])

    # Filter task_execution events
    task_events = [e for e in trace_events if e.get("cat") == "task_execution"]

    # Group by task name and match begin/end events
    task_dict = {}
    for event in task_events:
        task_name = event.get("name", "")
        phase = event.get("ph", "")
        ts = event.get("ts", 0)

        if task_name not in task_dict:
            task_dict[task_name] = {}

        if phase == "b":
            task_dict[task_name]["start"] = ts
        elif phase == "e":
            task_dict[task_name]["end"] = ts

    # Calculate durations
    latencies = []
    for task_name, times in task_dict.items():
        if "start" in times and "end" in times:
            duration_us = times["end"] - times["start"]
            duration_s = duration_us / 1e6  # Convert microseconds to seconds
            latencies.append(duration_s)

    return latencies


def get_dask_job_start_offset(log_dir):
    """
    Calculate the time offset from job start to task execution start for Dask.
    Reads main.log and finds the time between first timestamp and "Running X CPU tasks" line.

    Parameters:
    -----------
    log_dir : str
        Directory containing main.log

    Returns:
    --------
    float
        Time offset in seconds, or 0 if cannot calculate
    """
    main_log = os.path.join(log_dir, "main.log")
    if not os.path.exists(main_log):
        return 0.0

    try:
        from datetime import datetime

        first_timestamp = None
        running_timestamp = None

        with open(main_log, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse timestamp from line (format: 2026-01-28 04:06:09,491 - INFO - ...)
                try:
                    timestamp_str = line.split(" - ")[0].strip()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

                    if first_timestamp is None:
                        first_timestamp = timestamp

                    # Look for "Running X CPU tasks" or "Running X tasks" line
                    if (
                        "Running" in line
                        and "tasks" in line
                        and (
                            "CPU" in line or "GPU" in line or line.count("Running") == 1
                        )
                    ):
                        running_timestamp = timestamp
                        break
                except (ValueError, IndexError):
                    continue

        if first_timestamp and running_timestamp:
            offset = (running_timestamp - first_timestamp).total_seconds()
            return offset
    except Exception as e:
        print(f"Warning: Could not calculate Dask job start offset: {e}")

    return 0.0


def get_parsl_job_start_offset(log_dir):
    """
    Calculate the time offset from job start to task execution start for Parsl.
    Reads main.log and finds the time between first timestamp and "Running X CPU tasks" line.

    Parameters:
    -----------
    log_dir : str
        Directory containing main.log

    Returns:
    --------
    float
        Time offset in seconds, or 0 if cannot calculate
    """
    main_log = os.path.join(log_dir, "main.log")
    if not os.path.exists(main_log):
        return 0.0

    try:
        from datetime import datetime

        first_timestamp = None
        running_timestamp = None

        with open(main_log, "r") as f:
            for line in f:
                if not line.strip():
                    continue

                # Parse timestamp from line
                try:
                    timestamp_str = line.split(" - ")[0].strip()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

                    if first_timestamp is None:
                        first_timestamp = timestamp

                    # Look for "Running X CPU tasks" or "Running X tasks" line
                    if (
                        "Running" in line
                        and "tasks" in line
                        and (
                            "CPU" in line or "GPU" in line or line.count("Running") == 1
                        )
                    ):
                        running_timestamp = timestamp
                        break
                except (ValueError, IndexError):
                    continue

        if first_timestamp and running_timestamp:
            offset = (running_timestamp - first_timestamp).total_seconds()
            return offset
    except Exception as e:
        print(f"Warning: Could not calculate Parsl job start offset: {e}")

    return 0.0


def get_flux_job_start_offsets(log_dir):
    """
    Calculate time offsets for each Flux timeline file relative to the earliest job start.

    Algorithm:
    1. Check all *_main.log files and find the first timestamp
    2. Find the timestamp of the first "Submitting X tasks to URI Y" line in each file
    3. Identify the minimum first timestamp across all files
    4. The offset for each file is the difference between min timestamp and the Submitting timestamp

    Args:
        log_dir: Directory containing *_main.log files

    Returns:
        dict: Dictionary mapping timeline base name (e.g., 'x4316c7s6b0n0') -> offset in seconds
    """
    import glob
    import os
    import re
    from datetime import datetime

    # Find all *_main.log files
    main_log_files = glob.glob(os.path.join(log_dir, "*_main.log"))

    if not main_log_files:
        return {}

    # Store first timestamp and submitting timestamp for each file
    file_data = {}

    for log_file in main_log_files:
        base_name = os.path.basename(log_file).replace("_main.log", "")

        with open(log_file, "r") as f:
            lines = f.readlines()

        first_timestamp = None
        submitting_timestamp = None

        # Pattern for "Submitting X tasks to URI Y/Z" line
        submitting_pattern = re.compile(r"Submitting \d+ tasks to URI \d+/\d+")

        for line in lines:
            if not line.strip():
                continue

            try:
                # Parse timestamp from line (format: YYYY-MM-DD HH:MM:SS,fff)
                timestamp_str = line.split(" - ")[0].strip()
                timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")

                if first_timestamp is None:
                    first_timestamp = timestamp

                # Look for first "Submitting" line
                if submitting_timestamp is None and submitting_pattern.search(line):
                    submitting_timestamp = timestamp
                    break
            except (ValueError, IndexError):
                continue

        if first_timestamp and submitting_timestamp:
            file_data[base_name] = {
                "first_timestamp": first_timestamp,
                "submitting_timestamp": submitting_timestamp,
            }

    if not file_data:
        return {}

    # Find the minimum first timestamp across all files
    min_first_timestamp = min(data["first_timestamp"] for data in file_data.values())

    # Calculate offset for each file
    offsets = {}
    for base_name, data in file_data.items():
        # Offset is the difference between min first timestamp and this file's submitting timestamp
        offset = (data["submitting_timestamp"] - min_first_timestamp).total_seconds()
        offsets[base_name] = offset

    return offsets


def get_el_worker_offsets(log_dir):
    """
    Calculate time offsets for each EL worker relative to master start time.

    Algorithm:
    1. Find first timestamp from master-main.log
    2. For each worker-*.log file, find "Connected to parent" timestamp
    3. Calculate offset = worker_connected_timestamp - master_first_timestamp
    4. Store offset keyed by worker identifier (part between "worker-" and ".log")

    Args:
        log_dir: Directory containing master-main.log and worker-*.log files

    Returns:
        dict: Dictionary mapping worker_id -> offset in seconds
    """
    import re
    from datetime import datetime

    # Step 1: Get first timestamp from master-main.log
    master_log = os.path.join(log_dir, "master-main.log")
    if not os.path.exists(master_log):
        raise FileNotFoundError(f"Master log not found: {master_log}")

    with open(master_log, "r") as f:
        first_line = f.readline().strip()

    # Parse timestamp from first line
    # Format: 2026-01-27 00:31:34,369 - ...
    timestamp_match = re.match(
        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", first_line
    )
    if not timestamp_match:
        raise ValueError(f"Could not parse timestamp from master log: {first_line}")

    master_time_str = timestamp_match.group(1)
    master_time = datetime.strptime(master_time_str, "%Y-%m-%d %H:%M:%S,%f")

    # Step 2: Find all worker logs and extract "Connected to parent" timestamps
    worker_offsets = {}
    worker_pattern = os.path.join(log_dir, "worker-*.log")
    worker_files = glob.glob(worker_pattern)

    for worker_file in worker_files:
        # Extract worker_id from filename (between "worker-" and ".log")
        filename = os.path.basename(worker_file)
        worker_id = filename.replace("worker-", "").replace(".log", "")

        # Find "Connected to parent" line
        with open(worker_file, "r") as f:
            for line in f:
                if "Connected to parent" in line:
                    # Parse timestamp from this line
                    timestamp_match = re.match(
                        r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})", line
                    )
                    if timestamp_match:
                        worker_time_str = timestamp_match.group(1)
                        worker_time = datetime.strptime(
                            worker_time_str, "%Y-%m-%d %H:%M:%S,%f"
                        )

                        # Calculate offset in seconds
                        offset = (worker_time - master_time).total_seconds()
                        worker_offsets[worker_id] = offset
                    break

    return worker_offsets


# ============================================================================
# SCALING LOG READERS
# ============================================================================


def read_framework_log(log_file):
    """
    Read a Dask or Parsl log file and extract elapsed time.

    Looks for:
      - "Task submission completed in X.XX seconds"
      - "tasks finished in X.XX seconds"

    Returns:
        list: Single element with total elapsed time (submission + execution)
    """
    submission_time = None
    execution_time = None

    with open(log_file, "r") as f:
        for line in f:
            if "Task submission completed in" in line and "seconds" in line:
                parts = line.split()
                if parts[-1] == "seconds":
                    try:
                        submission_time = float(parts[-2])
                    except (ValueError, IndexError):
                        pass
            elif "tasks finished in" in line and "seconds" in line:
                parts = line.split()
                if parts[-1] == "seconds":
                    try:
                        execution_time = float(parts[-2])
                    except (ValueError, IndexError):
                        pass

    if submission_time is not None and execution_time is not None:
        return [submission_time + execution_time]
    elif execution_time is not None:
        return [execution_time]
    return []


def read_mpi_log(log_file):
    """
    Read MPI elapsed_time file and extract elapsed times.

    Looks for: "Elapsed time for test: X.XXX seconds"

    Returns:
        list: Elapsed times extracted from log
    """
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    elapsed_times = []
    with open(log_file, "r") as f:
        for line in f:
            if "Elapsed time for test:" in line and "seconds" in line:
                parts = line.split()
                if parts[-1] == "seconds":
                    try:
                        elapsed_times.append(float(parts[-2]))
                    except (ValueError, IndexError):
                        pass
    return elapsed_times


def read_el_log(log_dir):
    """
    Extract work_time from EL main.log.

    Returns:
        list: [work_time] in seconds
    """
    log_file = os.path.join(log_dir, "main.log")

    if not os.path.exists(log_file):
        raise FileNotFoundError(f"Log file not found: {log_file}")

    with open(log_file, "r") as f:
        lines = f.readlines()

    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    first_timestamp = None
    first_task_update_timestamp = None
    all_children_done_timestamp = None

    for line in lines:
        match = re.match(timestamp_pattern, line)
        if not match:
            continue
        timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
        if first_timestamp is None:
            first_timestamp = timestamp
        if "Sent initial task update" in line and first_task_update_timestamp is None:
            first_task_update_timestamp = timestamp
        if "All children have completed execution" in line:
            all_children_done_timestamp = timestamp

    if not all(
        [first_timestamp, first_task_update_timestamp, all_children_done_timestamp]
    ):
        raise ValueError(f"Could not find all required log markers in {log_file}")

    work_time = (
        all_children_done_timestamp - first_task_update_timestamp
    ).total_seconds()
    return [work_time]


def read_el_cluster_log(logdir: str):
    fname = os.path.join(logdir, "script.log")
    with open(fname, "r") as f:
        lines = f.readlines()
        for line in lines:
            if "tasks finished in" in line and "seconds" in line:
                parts = line.split()
                if parts[-1] == "seconds":
                    try:
                        execution_time = float(parts[-2])
                    except (ValueError, IndexError):
                        pass
    return [execution_time]


def read_flux_log(log_dir):
    """
    Extract elapsed time from Flux *_main.log files.

    Returns:
        list: [elapsed_time] in seconds
    """
    log_files = glob.glob(os.path.join(log_dir, "*_main.log"))
    if not log_files:
        raise FileNotFoundError(f"No *_main.log files found in: {log_dir}")

    timestamp_pattern = r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})"
    min_submitting_timestamp = None
    max_finished_timestamp = None

    for log_file in log_files:
        with open(log_file, "r") as f:
            for line in f:
                match = re.match(timestamp_pattern, line)
                if not match:
                    continue
                timestamp = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S,%f")
                if "Submitting" in line and "tasks to URI" in line:
                    if (
                        min_submitting_timestamp is None
                        or timestamp < min_submitting_timestamp
                    ):
                        min_submitting_timestamp = timestamp
                if "All" in line and "tasks finished" in line:
                    if (
                        max_finished_timestamp is None
                        or timestamp > max_finished_timestamp
                    ):
                        max_finished_timestamp = timestamp

    if min_submitting_timestamp is None or max_finished_timestamp is None:
        raise ValueError(f"Could not find required log markers in {log_dir}")

    return [(max_finished_timestamp - min_submitting_timestamp).total_seconds()]


def determine_system_type(directory):
    """
    Determine system type from directory path.

    Returns:
        str: 'dask', 'parsl', 'mpi', 'flux', or 'el'
    """
    dir_lower = directory.lower()
    if "dask" in dir_lower:
        return "dask"
    elif "parsl" in dir_lower:
        return "parsl"
    elif "mpi" in dir_lower:
        return "mpi"
    elif "flux" in dir_lower:
        return "flux"
    elif "el_cluster" in dir_lower:
        if "ws" in dir_lower:
            return "el_cluster_pull"
        return "el_cluster"
    return "el"


def read_plot_data_from_csv(csv_file):
    """
    Read plot data from CSV file.

    Returns:
        dict or None: system name -> {'nodes', 'mean', 'std'}, or None if file missing
    """
    if not os.path.exists(csv_file):
        return None

    plot_data = {}
    with open(csv_file, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # skip metadata line
        headers = next(reader)
        systems = []
        for i in range(1, len(headers), 2):
            system_name = headers[i].replace("_mean", "")
            systems.append(system_name)
            plot_data[system_name] = {"nodes": [], "mean": [], "std": []}
        for row in reader:
            node_count = int(row[0])
            for i, system in enumerate(systems):
                mean_idx = 1 + i * 2
                std_idx = 2 + i * 2
                if row[mean_idx] and row[std_idx]:
                    plot_data[system]["nodes"].append(node_count)
                    plot_data[system]["mean"].append(float(row[mean_idx]))
                    plot_data[system]["std"].append(float(row[std_idx]))
    return plot_data


def save_plot_data_to_csv(plot_data, csv_file, sleep_time, nworkers, ntasks_per_worker):
    """
    Save plot data to CSV file.
    """
    all_nodes = sorted({node for data in plot_data.values() for node in data["nodes"]})

    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                f"# sleep_time={sleep_time}ms, nworkers={nworkers}, ntasks_per_worker={ntasks_per_worker}"
            ]
        )
        headers = ["nodes"]
        for system in sorted(plot_data.keys()):
            headers.extend([f"{system}_mean", f"{system}_std"])
        writer.writerow(headers)

        for node in all_nodes:
            row = [node]
            for system in sorted(plot_data.keys()):
                data = plot_data[system]
                if node in data["nodes"]:
                    idx = data["nodes"].index(node)
                    row.extend([data["mean"][idx], data["std"][idx]])
                else:
                    row.extend(["", ""])
            writer.writerow(row)


def apply_timeline_offset(csv_file, offset):
    """
    Apply a time offset to completion_time in a timeline CSV.
    Creates a modified dataframe with adjusted times.

    First makes completion_time relative to the start, then adds the offset.

    Parameters:
    -----------
    csv_file : str
        Path to the timeline CSV file
    offset : float
        Time offset in seconds to add to completion_time

    Returns:
    --------
    pd.DataFrame
        DataFrame with adjusted completion_time
    """
    df = pd.read_csv(csv_file)
    if "completion_time" in df.columns:
        # First make times relative to start
        start_time = df["completion_time"].min()
        df["relative_time"] = df["completion_time"] - start_time

        print(df["completion_time"])
        # Then add the offset
        if offset > 0:
            df["relative_time"] = df["relative_time"] + offset
    return df


def get_flux_client_timeline(dirname):
    files = glob.glob(f"{dirname}/*_timeline.csv")
    dfs = []
    for fname in files:
        dfs.append(pd.read_csv(fname))
    df_merged = pd.concat(dfs)

    return df_merged


def get_trigger_time(dirname):
    fname = os.path.join(dirname, "main.log")
    if "parsl" in dirname.lower():
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                ###When the runs are partially run then there will be multiple trigger points. We want the latest
                if "Triggering task execution..." in line:
                    timestamp_str = line.split(" - ")[0].strip()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
        return timestamp.timestamp()
    elif "dask" in dirname.lower():
        with open(fname, "r") as f:
            lines = f.readlines()
            for line in lines:
                "2026-02-14 17:41:24,265 - INFO - Writing trigger file at timestamp 1771090884.265515089..."
                written_times = []
                ###When the runs are partially run then there will be multiple trigger points. We want the latest
                if "Writing trigger file at timestamp" in line:
                    timestamp_str = line.split(" - ")[0].strip()
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S,%f")
                    computed_time = timestamp.timestamp()
                    written_time = float(line.split()[-1][:-3])
                    print(
                        f"Differencnce between computed and written:{abs(computed_time - written_time)}"
                    )
        return written_time


# scheduling latency is defined as the difference between the submission time on the client side and worker side
def compute_scheduling_latency(data_dir):
    fname_worker = os.path.join(data_dir, "timeline_worker.csv")
    df_worker = pd.read_csv(fname_worker)

    if "flux" in data_dir:
        df_client = get_flux_client_timeline(data_dir)
    else:
        if "trigger" not in data_dir:
            fname_client = os.path.join(data_dir, "timeline_client.csv")
            if not os.path.exists(fname_client):
                fname_client = os.path.join(data_dir, "timeline_CPU.csv")
            df_client = pd.read_csv(fname_client)
        else:
            print(f"Using trigger mechanism for {data_dir}")
            trigger_time = get_trigger_time(data_dir)
            n_rows = len(pd.read_csv(fname_worker))
            df_client = pd.DataFrame(
                {"start_time(s)": [trigger_time] * n_rows, "task_id": range(n_rows)}
            )

    # Filter out negative task_ids
    df_client = df_client[df_client["task_id"] >= 0]
    df_worker = df_worker[df_worker["task_id"] >= 0]

    # Merge dataframes on task_id
    df_merged = pd.merge(
        df_client, df_worker, on="task_id", suffixes=("_client", "_worker")
    )

    # Calculate scheduling latency (client start time - worker start time)
    scheduling_latencies = (
        df_merged["start_time(s)_worker"] - df_merged["start_time(s)_client"]
    )

    return scheduling_latencies.tolist()


def compute_completion_latency(data_dir):
    if "flux" in data_dir:
        df_client = get_flux_client_timeline(data_dir)
    else:
        fname_client = os.path.join(data_dir, "timeline_client.csv")
        if not os.path.exists(fname_client):
            fname_client = os.path.join(data_dir, "timeline_CPU.csv")
        df_client = pd.read_csv(fname_client)
    fname_worker = os.path.join(data_dir, "timeline_worker.csv")
    df_worker = pd.read_csv(fname_worker)

    # Filter out negative task_ids
    df_client = df_client[df_client["task_id"] >= 0]
    df_worker = df_worker[df_worker["task_id"] >= 0]

    # Merge dataframes on task_id
    df_merged = pd.merge(
        df_client, df_worker, on="task_id", suffixes=("_client", "_worker")
    )

    # Calculate scheduling latency (client start time - worker start time)
    completion_latencies = (
        df_merged["end_time(s)_client"] - df_merged["end_time(s)_worker"]
    )

    return completion_latencies.tolist()
