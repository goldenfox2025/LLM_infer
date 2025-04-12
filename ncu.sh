#!/bin/bash
set -e

echo ">> Starting Nsight Compute profiling..."

# 记录原始目录 (父目录)
original_dir=$(pwd)
echo ">> Original directory: ${original_dir}"

# --- New: Define and ensure the 'data' directory exists ---
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
# --- End New ---

# 设置 perf_event_paranoid (保持之前的健壮性检查)
if [ -w /proc/sys/kernel/perf_event_paranoid ]; then
    if sudo -n sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid' 2>/dev/null; then
        echo ">> Set perf_event_paranoid to 1."
    else
        echo ">> Warning: Could not set perf_event_paranoid without interaction, attempting without sudo..."
        if sh -c 'echo 1 > /proc/sys/kernel/perf_event_paranoid' 2>/dev/null; then
           echo ">> Set perf_event_paranoid to 1 (without sudo)."
        else
           echo ">> Warning: Could not write to /proc/sys/kernel/perf_event_paranoid. Profiling might be limited if kernel events are needed."
        fi
    fi
elif [ -e /proc/sys/kernel/perf_event_paranoid ]; then
     echo ">> Warning: No write permission for /proc/sys/kernel/perf_event_paranoid. Profiling might be limited if kernel events are needed."
fi


# 检查 frontend 目录是否存在于当前 (父) 目录
if [ -d "frontend" ]; then
    # 定义输出报告文件名 (包含时间戳)
    report_filename="chat_profile_$(date +%Y%m%d_%H%M%S).ncu-rep"
    # 定义报告文件在 'data' 目录中的完整路径
    report_full_path="${data_dir_path}/${report_filename}"

    # 切换到 frontend 目录
    cd frontend
    echo ">> Changed directory to $(pwd)."

    echo ">> Running Nsight Compute on chat.py..."
    # *** 关键修改：指定输出路径为父目录下的 'data' 目录 ***
    # 使用相对路径 "../data/" 指向原始目录下的 data 文件夹
    echo ">> Report file will be saved to: ${report_full_path}"
    ncu \
      -o "../${data_dir_name}/${report_filename}" \
      --set full \
      --target-processes all \
      --replay-mode kernel \
      python3 chat.py
      # 如果 chat.py 需要参数，在这里加上: python3 chat.py arg1 arg2 ...

    ncu_exit_status=$? # 保存 ncu 的退出状态

    # *** 返回上一级目录 (原始目录) ***
    cd "${original_dir}" # 使用保存的原始目录路径更安全
    echo ">> Returned to directory: $(pwd)"

    # 根据 ncu 退出状态进行判断和报告
    if [ $ncu_exit_status -eq 0 ]; then
        echo ">> Nsight Compute profiling finished successfully."
        # 报告已经在父目录的 data 子目录中，路径正确
        echo ">> Report saved to: ${report_full_path}"
        # 提供查看命令，可以使用相对路径 data/ 或完整路径
        echo ">> You can view the report using the Nsight Compute GUI: ncu-ui \"${report_full_path}\""
        # 或者: echo ">> You can view the report using the Nsight Compute GUI: ncu-ui \"${data_dir_name}/${report_filename}\""
    else
        echo ">> Nsight Compute profiling failed with exit code ${ncu_exit_status}."
        # 报告文件可能未生成或不完整，但仍提示预期路径
        echo ">> Expected report location: ${report_full_path}"
        exit 1 # 以失败状态退出
    fi

else
    echo ">> Error: frontend directory not found in ${original_dir}!"
    # 已经在父目录，直接退出
    exit 1
fi

# 可选：恢复 perf_event_paranoid 设置
# ...

exit 0