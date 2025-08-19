#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.

echo ">> Starting Nsight Compute profiling (with --set full)..."
echo ">> Ensure this is run from a TTY (Ctrl+Alt+F3) or SSH to avoid desktop freeze."

# Record the original directory (parent directory)
original_dir=$(pwd)
echo ">> Original directory: ${original_dir}"

# --- Define and ensure the 'data' directory exists ---
data_dir_name="data"
data_dir_path="${original_dir}/${data_dir_name}"

if [ ! -d "${data_dir_path}" ]; then
    echo ">> Data directory '${data_dir_path}' not found. Creating it..."
    mkdir -p "${data_dir_path}"
    if [ $? -ne 0 ]; then
        echo ">> Error: Failed to create data directory '${data_dir_path}'!"
        exit 1
    fi
    echo ">> Created data directory: ${data_dir_path}"
else
    echo ">> Using existing data directory: ${data_dir_path}"
fi
# --- End Data Dir ---

# --- Set perf_event_paranoid (using robust check) ---
paranoid_path="/proc/sys/kernel/perf_event_paranoid"
current_paranoid_value=$(cat "$paranoid_path" 2>/dev/null || echo "unknown")
echo ">> Current perf_event_paranoid value: ${current_paranoid_value}"

# Attempt to set only if it's currently > 1
if [ "$current_paranoid_value" != "unknown" ] && [ "$current_paranoid_value" -gt 1 ]; then
    echo ">> Attempting to set perf_event_paranoid to 1..."
    # Try direct write first (if running as root or have direct permission)
    if echo 1 > "$paranoid_path" 2>/dev/null; then
        echo ">> Successfully set perf_event_paranoid to 1 (direct write)."
    # If direct write fails, try sudo without password prompt
    elif sudo -n sh -c "echo 1 > $paranoid_path" 2>/dev/null; then
        echo ">> Successfully set perf_event_paranoid to 1 using sudo -n."
    else
        echo ">> Warning: Could not set perf_event_paranoid to 1 automatically."
        echo ">> Profiling might be limited if kernel events are needed."
        echo ">> Consider running 'sudo sysctl kernel.perf_event_paranoid=1' manually if issues arise."
        # Don't exit, let ncu try anyway
    fi
elif [ "$current_paranoid_value" == "unknown" ]; then
    echo ">> Warning: Could not read $paranoid_path. Cannot check or set paranoia level."
else
    echo ">> perf_event_paranoid is already at 1 or lower. No change needed."
fi
# --- End perf_event_paranoid ---

# Check if frontend directory exists in the current (parent) directory
if [ -d "frontend" ]; then
    # Define output report filename (including timestamp)
    timestamp=$(date +%Y%m%d_%H%M%S)
    report_filename="chat_profile_${timestamp}_full.ncu-rep" # Added _full suffix
    # Define full path for the report file within the 'data' directory
    report_full_path="${data_dir_path}/${report_filename}"

    # Change to the frontend directory

    echo ">> Changed directory to $(pwd)."

    echo ">> Running Nsight Compute on chat.py with --set full for kernel 'gemmv'..."
    echo ">> This WILL take a long time and consume significant resources."

    # Construct the ncu command arguments
    # Using the options that caused the freeze in the GUI, but should work in TTY/SSH
    ncu_args=(
    ncu
    -o "./${data_dir_name}/${report_filename}"
    --set full
    --target-processes all
    --replay-mode kernel
    --kernel-name 'awq_gemm_kernel_mma'
    python3 frontend/chat.py
    # ... chat.py 的参数 ...
    )


    echo ">> Executing: ${ncu_args[*]}"

    # Execute ncu command
    "${ncu_args[@]}"

    ncu_exit_status=$? # Save ncu's exit status

    # Return to the original directory
    cd "${original_dir}"
    echo ">> Returned to directory: $(pwd)"

    # Report based on ncu exit status
    if [ $ncu_exit_status -eq 0 ]; then
        echo ">> Nsight Compute profiling (--set full) finished successfully."
        echo ">> Report saved to: ${report_full_path}"
        echo ">> You can view the report later using the Nsight Compute GUI:"
        echo "   ncu-ui \"${report_full_path}\""
    else
        echo ">> Nsight Compute profiling failed with exit code ${ncu_exit_status}."
        echo ">> Check the output above for specific errors from ncu or the application."
        echo ">> Expected report location (may be incomplete or not generated): ${report_full_path}"
        exit 1 # Exit with failure status
    fi

else
    echo ">> Error: frontend directory not found in ${original_dir}!"
    # Already in the parent directory, just exit
    exit 1
fi

echo ">> Script finished."
exit 0