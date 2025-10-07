env BITUNIX_API_KEY=<> BITUNIX_API_SECRET=<> PYTHONPATH=/Users/mac/Downloads/sources/scalp-test python3 scripts/bitunix_update_stop_loss.py --order-id 2036384585239822336 --stop-price 2150.50 --qty 0.003

env BITUNIX_API_KEY=<> BITUNIX_API_SECRET=<> PYTHONPATH=/Users/mac/Downloads/sources/scalp-test python3 scripts/bitunix_list_positions.py --symbol ETHUSDT

PYTHONPATH=/Users/mac/Downloads/sources/scalp-test python3 scripts/bitunix_modify_position_tpsl.py --symbol ETHUSDT --position-id 8212749670516344550 --sl-price 1000 --sl-stop-type MARK_PRICE --key <> --secret <>

env BITUNIX_API_KEY=<> BITUNIX_API_SECRET=<> PYTHONPATH=/Users/mac/Downloads/sources/scalp-test  python3 scripts/bitunix_place_position_tpsl.py --symbol ETHUSDT --position-id 1324379971231393707 --sl-price 1000 --sl-stop-type MARK_PRICE
