#!/bin/bash
set -e

echo ">> Starting Nsight Systems profiling with GPU info..."

# 记录原始目录 (父目录)
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
# --- End New ---

# 设置 perf_event_paranoid (确保能够进行完整的性能分析)
# 降低内核 perf_event_paranoid 参数（需要 sudo 权限）
if [ -e /proc/sys/kernel/perf_event_paranoid ]; then
    current_value=$(cat /proc/sys/kernel/perf_event_paranoid)
    echo ">> Current perf_event_paranoid value: $current_value"

    # 尝试设置为-1（最低限制，允许所有性能分析功能）
    if [ -w /proc/sys/kernel/perf_event_paranoid ]; then
        if sudo -n sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid' 2>/dev/null; then
            echo ">> Successfully set perf_event_paranoid to -1 (full profiling enabled)."
        else
            echo ">> Attempting to set perf_event_paranoid with interactive sudo..."
            echo ">> Please enter your password if prompted:"
            if sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'; then
                echo ">> Successfully set perf_event_paranoid to -1 (full profiling enabled)."
            else
                echo ">> Warning: Failed to set perf_event_paranoid. Some profiling features may be limited."
                echo ">> To enable full profiling, run: sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'"
            fi
        fi
    else
        echo ">> Warning: No write permission for /proc/sys/kernel/perf_event_paranoid."
        echo ">> To enable full profiling, run: sudo sh -c 'echo -1 > /proc/sys/kernel/perf_event_paranoid'"
        echo ">> Continuing with limited profiling capabilities..."
    fi
else
    echo ">> Note: perf_event_paranoid not found. This might not be a Linux system or the kernel doesn't support this feature."
fi

# （可选）激活 conda 环境 - 保持注释状态，用户可根据需要启用
# source /home/keta/miniconda3/bin/activate myenv

# 检查 frontend 目录是否存在于当前 (父) 目录
if [ -d "frontend" ]; then
    # 定义输出报告文件名 (包含时间戳, nsys 默认扩展名通常是 .qdrep)
    report_base_name="nsys_profile_report"
    timestamp=$(date +%Y%m%d_%H%M%S)
    report_filename="${report_base_name}_${timestamp}" # nsys 会自动添加 .qdrep
    # 定义报告文件在 'data' 目录中的完整路径
    report_full_path="${data_dir_path}/${report_filename}.qdrep" # 显式添加扩展名以便引用

    # 确保data目录存在并有写入权限
    mkdir -p "${data_dir_path}"
    if [ ! -w "${data_dir_path}" ]; then
        echo ">> Warning: No write permission for ${data_dir_path}. Will use /tmp instead."
        data_dir_path="/tmp"
        report_full_path="${data_dir_path}/${report_filename}.qdrep"
    fi

    # 切换到 frontend 目录
    # cd frontend
    echo ">> Changed directory to $(pwd)."

    echo ">> Running Nsight Systems on chat.py..."
    # *** 关键修改：指定输出路径为父目录下的 'data' 目录，并使用带时间戳的文件名 ***
    echo ">> Report file will be saved to: ${report_full_path}"
    # 确定输出路径
    if [ "${data_dir_path}" = "/tmp" ]; then
        output_path="${data_dir_path}/${report_filename}"
    else
        output_path="./${data_dir_name}/${report_filename}"
    fi

    echo ">> Using output path: ${output_path}"

    nsys profile \
      --trace=cuda,osrt,nvtx,cudnn,cublas,oshmem \
      --gpu-metrics-devices=0 \
      --sample=cpu \
      --backtrace=dwarf \
      --cpuctxsw=process-tree \
      --stats=true \
      --export=sqlite \
      --output "${output_path}" \
      --force-overwrite true \
      python3 frontend/chat.py
      # 如果 chat.py 需要参数，在这里加上: python3 chat.py arg1 arg2 ...

    nsys_exit_status=$? # 保存 nsys 的退出状态

    # *** 返回上一级目录 (原始目录) ***
    cd "${original_dir}" # 使用保存的原始目录路径更安全
    echo ">> Returned to directory: $(pwd)"

    # 根据 nsys 退出状态进行判断和报告
    if [ $nsys_exit_status -eq 0 ]; then
        echo ">> Nsight Systems profiling finished successfully."

        # 检查报告文件是否存在
        qdrep_file="${output_path}.qdrep"
        sqlite_file="${output_path}.sqlite"

        # 检查.qdrep文件
        if [ -f "${qdrep_file}" ]; then
            echo ">> Report saved to: ${qdrep_file}"
            echo ">> You can view the report using the Nsight Systems GUI: nsys-ui \"${qdrep_file}\""

            # 复制报告到用户主目录，确保可以访问
            user_home_copy="${HOME}/nsys_latest_report.qdrep"
            cp "${qdrep_file}" "${user_home_copy}"
            echo ">> A copy of the report has been saved to: ${user_home_copy}"
            echo ">> You can also view it with: nsys-ui \"${user_home_copy}\""
        else
            echo ">> Warning: Could not find .qdrep file at expected location: ${qdrep_file}"

            # 查找可能的报告文件
            tmp_reports=$(find /tmp -name "nsys-report-*.nsys-rep" -type f -mmin -5 2>/dev/null)
            if [ -n "${tmp_reports}" ]; then
                echo ">> Found recent report files in /tmp:"
                echo "${tmp_reports}"
                first_report=$(echo "${tmp_reports}" | head -n 1)
                echo ">> You can try viewing: nsys-ui \"${first_report}\""

                # 复制第一个找到的报告到用户主目录
                user_home_copy="${HOME}/nsys_latest_report.nsys-rep"
                cp "${first_report}" "${user_home_copy}"
                echo ">> A copy of the report has been saved to: ${user_home_copy}"
                echo ">> You can also view it with: nsys-ui \"${user_home_copy}\""
            fi
        fi

        # 检查.sqlite文件
        if [ -f "${sqlite_file}" ]; then
            echo ">> SQLite database saved to: ${sqlite_file}"

            # 复制SQLite文件到用户主目录
            user_home_copy="${HOME}/nsys_latest_report.sqlite"
            cp "${sqlite_file}" "${user_home_copy}"
            echo ">> A copy of the SQLite database has been saved to: ${user_home_copy}"
        fi
    else
        echo ">> Nsight Systems profiling failed with exit code ${nsys_exit_status}."
        # 报告文件可能未生成或不完整，但仍提示预期路径
        echo ">> Expected report location: ${output_path}.qdrep"
        exit 1 # 以失败状态退出
    fi

else
    echo ">> Error: frontend directory not found in ${original_dir}!"
    # 已经在父目录，直接退出
    exit 1
fi

# 恢复 perf_event_paranoid 设置（如果之前修改过）
if [ -e /proc/sys/kernel/perf_event_paranoid ] && [ -n "$current_value" ]; then
    echo ">> Restoring perf_event_paranoid to original value: $current_value"
    if [ -w /proc/sys/kernel/perf_event_paranoid ]; then
        if sudo -n sh -c "echo $current_value > /proc/sys/kernel/perf_event_paranoid" 2>/dev/null; then
            echo ">> Successfully restored perf_event_paranoid to $current_value."
        else
            echo ">> Note: Could not automatically restore perf_event_paranoid. This is not critical."
        fi
    fi
fi

echo ">> Nsight Systems profiling completed."
exit 0