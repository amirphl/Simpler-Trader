#!/usr/bin/env fish
# Start one asynchronous EMA + AVWAP coordinator for each configured live account.

set -l script_dir (cd (dirname (status filename)); and pwd)
set -l project_root (cd "$script_dir/.."; and pwd)
cd "$project_root"; or exit 1

set -l python_bin "$project_root/.venv/bin/python"
if not test -x "$python_bin"
    echo "Error: expected Python environment at $python_bin" >&2
    exit 1
end

# Live entry/exit evaluation uses Bitunix's forming-candle WebSocket stream.
# Keep this check before confirmation/backgrounding: a stale virtualenv used to
# let all accounts start and then silently fail closed for every live signal.
if not "$python_bin" -c "import websocket" >/dev/null 2>&1
    echo "Error: websocket-client is missing from $python_bin" >&2
    echo "Install this checkout's dependencies, then start again:" >&2
    echo "  $python_bin -m pip install -r requirements.txt" >&2
    exit 1
end

# nohup disconnects stdin, so obtain explicit confirmation before backgrounding.
if test "$LIVE_TRADING_CONFIRM" != YES
    read -l -P "LIVE TRADING: type YES to start all three accounts: " response
    if test "$response" != YES
        echo "Live trading cancelled."
        exit 0
    end
end

function start_account --argument-names account
    # Fish functions do not inherit the caller's local variables. Define this
    # inside the function so nohup always receives the Python executable.
    set -l python_bin "$PWD/.venv/bin/python"
    set -l config_file "configs/live_trading.ema_avwap_pullback_$account.env"
    set -l data_root "./data/ema_avwap_pullback/account_$account"
    set -l log_root "./logs/ema_avwap_pullback/account_$account"
    set -l pid_root "$data_root/pids"
    set -l pid_file "$pid_root/coordinator.pid"
    set -l stdout_file "$log_root/stdout.log"

    if not test -f "$config_file"
        echo "[account $account] config not found: $config_file" >&2
        return 1
    end

    mkdir -p "$pid_root" "$log_root"; or return 1

    if test -f "$pid_file"
        set -l old_pid (string trim -- (cat "$pid_file"))
        if test -n "$old_pid"; and kill -0 "$old_pid" 2>/dev/null
            echo "[account $account] already running (PID $old_pid); skipping"
            return 0
        end
    end

    # API credentials are deliberately removed only for this child process so
    # the selected config file supplies that account's credentials.
    env -u API_KEY -u API_SECRET -u API_PASSPHRASE LIVE_TRADING_CONFIRM=YES nohup "$python_bin" -m cmd.live_trading.ema_avwap_pullback_main --config-file "$config_file" --symbols FETUSDT,SOLUSDT,ADAUSDT,ICPUSDT --state-file "$data_root/state.json" --positions-db "$data_root/positions.db" --klines-db "$data_root/klines.db" --log-file "$log_root/live.log" --live > "$stdout_file" 2>&1 &
    set -l pid $last_pid
    echo "$pid" > "$pid_file"; or return 1
    echo "[account $account] started (PID $pid)"
    echo "[account $account] stdout: $stdout_file"
end

set -l failures 0
for account in 1 2 3
    start_account "$account"; or set failures 1
end

exit $failures
