
# This script is meant to be sourced.
# It provides utilities to allocate and manage local GPU resources.

function is_available() {
    local i=$1
    local mem_gpu=$2

    # Query utilization, memory used, and pstate (to detect errors) in one call.
    local stats
    stats=$(nvidia-smi -i "$i" --query-gpu=utilization.gpu,memory.used,pstate --format=csv,noheader,nounits 2>/dev/null) || return 1

    local util=$(echo "$stats" | cut -d',' -f1 | tr -d ' ')
    local mem_used=$(echo "$stats" | cut -d',' -f2 | tr -d ' ')
    local pstate=$(echo "$stats" | cut -d',' -f3 | tr -d ' ')

    if [ -z "$util" ] || [ -z "$mem_used" ]; then
        return 1
    fi

    # CRITERIA CHECK:
    # - Is utilization 0?
    # - Is memory used < mem_gpu limit?
    # - Pstate should not be an error state (ignore empty just in case, but avoid anything that indicates a broken state, though usually pstate is Px. We can check if it's alphanumeric and format matches P*)
    if [ "$util" -eq 0 ] && [ "$mem_used" -lt "$mem_gpu" ] && [[ "$pstate" =~ P[0-9]+ ]]; then
        return 0
    fi
    return 1
}

function get_gpu() {
    local mem_gpu=${1:-16}
    local timeout_secs=${2:-0} # 0 means wait indefinitely

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | tail -n 1) || return 1
    if [ -z "$gpu_count" ]; then echo "No GPUs available"; return 1; fi

    local start_time=$(date +%s)

    while true; do
        # Loop through each GPU index
        for (( i=0; i<gpu_count; i++ ))
        do
            if is_available "$i" "$mem_gpu"; then
                echo $i
                return 0
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached"
                return 1
            fi
        fi
        sleep 5
    done
}

function get_gpus() {
    local num_gpus=${1:-1}
    local mem_gpu=${2:-16}
    local timeout_secs=${3:-0} # 0 means wait indefinitely

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | tail -n 1) || return 1
    if [ -z "$gpu_count" ]; then return 1; fi

    local start_time=$(date +%s)

    while true; do
        local idle_gpus=""
        local found=0

        # Loop through each GPU index
        for (( i=0; i<gpu_count; i++ ))
        do
            if is_available "$i" "$mem_gpu"; then
                if [ -z "$idle_gpus" ]; then
                    idle_gpus="$i"
                else
                    idle_gpus="${idle_gpus},$i"
                fi
                found=$((found + 1))
                if [ "$found" -eq "$num_gpus" ]; then
                    echo "$idle_gpus"
                    return 0
                fi
            fi
        done

        if [ "$timeout_secs" -gt 0 ]; then
            local current_time=$(date +%s)
            if [ $((current_time - start_time)) -ge "$timeout_secs" ]; then
                echo "Timeout reached"
                return 1
            fi
        fi
        sleep 5
    done
}

# lock_gpu <gpu_id>
# Allocates a small amount of memory on the given GPU to "lock" it.
# This prevents other scripts using get_gpu from selecting it.
# It starts a background process and prints its PID.
# The caller is responsible for killing this PID to unlock the GPU.
# Requires python installed.
function lock_gpu() {
    local gpu_id=$1
    if [ -z "$gpu_id" ]; then
        echo "Invalid GPU ID: argument is empty" >&2
        return 1
    fi
    if [ "$gpu_id" -lt 0 ]; then
        echo "Invalid GPU ID: $gpu_id" >&2
        return 1
    fi
    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    local ready_file
    ready_file=$(mktemp)

    python "$script_dir/lock_gpu.py" "$gpu_id" > "$ready_file" 2>/dev/null &
    local lock_pid=$!

    # Poll until we see 'READY' or the process dies (max ~10s).
    local ready=0
    for (( i=0; i<40; i++ )); do
        if ! ps -p "$lock_pid" >/dev/null 2>&1; then
            break
        fi
        if grep -q "READY" "$ready_file" 2>/dev/null; then
            ready=1
            break
        fi
        sleep 0.25
    done
    rm -f "$ready_file"

    if [ "$ready" -eq 1 ]; then
        echo "$lock_pid"
        return 0
    else
        kill -9 "$lock_pid" 2>/dev/null || true
        echo ""
        return 1
    fi
}

# unlock_gpu <pid>
# Kills the background process that is holding the GPU memory.
function unlock_gpu() {
    local pid=$1
    [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null
}

function lock_gpus() {
    local gpus_list=$1
    local pids=""
    IFS=',' read -ra gpus <<< "$gpus_list"
    for gpu in "${gpus[@]}"; do
        local pid
        pid=$(lock_gpu "$gpu")
        if [ -z "$pid" ]; then
            # Lock failed — roll back any GPUs locked so far
            if [ -n "$pids" ]; then
                unlock_gpus "$pids"
            fi
            echo "" >&2
            echo "Failed to lock GPU $gpu, rolled back previously locked GPUs." >&2
            return 1
        fi
        if [ -z "$pids" ]; then
            pids="$pid"
        else
            pids="${pids},${pid}"
        fi
    done
    echo "$pids"
}

function unlock_gpus() {
    local pids_list=$1
    IFS=',' read -ra pids <<< "$pids_list"
    for pid in "${pids[@]}"; do
        unlock_gpu "$pid"
    done
}
