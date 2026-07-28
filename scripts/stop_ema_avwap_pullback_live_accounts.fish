#!/usr/bin/env fish
# Gracefully stop asynchronous EMA + AVWAP account coordinators by saved PID.

set -l script_dir (cd (dirname (status filename)); and pwd)
set -l project_root (cd "$script_dir/.."; and pwd)
cd "$project_root"; or exit 1

set -l accounts 1 2 3
if test (count $argv) -gt 0
    set accounts $argv
end

set -l failures 0
for account in $accounts
    switch "$account"
        case 1 2 3
        case '*'
            echo "Unknown account '$account'; use 1, 2, or 3." >&2
            set failures 1
            continue
    end

    set -l pid_file "./data/ema_avwap_pullback/account_$account/pids/coordinator.pid"
    if not test -f "$pid_file"
        echo "[account $account] PID file not found: $pid_file"
        set failures 1
        continue
    end

    set -l pid (string trim -- (cat "$pid_file"))
    if test -z "$pid"
        echo "[account $account] empty PID file: $pid_file" >&2
        set failures 1
    else if kill -0 "$pid" 2>/dev/null
        echo "[account $account] stopping PID $pid"
        kill -TERM "$pid"; or set failures 1
    else
        echo "[account $account] not running (stale PID $pid)"
        set failures 1
    end
end

exit $failures
