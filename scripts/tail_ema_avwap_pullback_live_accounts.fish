#!/usr/bin/env fish
# Follow the stdout logs for all three asynchronous EMA + AVWAP coordinators.

set -l script_dir (cd (dirname (status filename)); and pwd)
set -l project_root (cd "$script_dir/.."; and pwd)
cd "$project_root"; or exit 1

tail -n 100 -F ./logs/ema_avwap_pullback/account_1/stdout.log ./logs/ema_avwap_pullback/account_2/stdout.log ./logs/ema_avwap_pullback/account_3/stdout.log
