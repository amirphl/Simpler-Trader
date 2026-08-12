# Debian deployment: Backtest panel and EMA + AVWAP

This bundle deploys the existing FastAPI backtest panel as user `debian` from
`/home/debian/Simpler-Trader`. Nginx is the only public listener; Uvicorn binds
to `127.0.0.1:9092`. The public EMA + AVWAP page is:

```text
https://jzbe.jazebeh.ir:15443/static/ema_avwap_pullback.html
```

The Nginx virtual host uses the existing certificate at
`/etc/letsencrypt/live/jzbe.jazebeh.ir/fullchain.pem` and `privkey.pem`, enables
TLS 1.2/1.3, proxies WebSockets, applies a request limit, and requires HTTP
Basic Authentication. Authentication matters here: submitting a backtest can
consume CPU, bandwidth, and database space.

## Before deployment

1. SSH to the server and update the checkout under `/home/debian/Simpler-Trader`.
   The deployer intentionally deploys the checked-out revision; it does not run
   `git pull` or change branches.
2. Confirm that DNS for `jzbe.jazebeh.ir` reaches this server and that the
   certificate files already exist at the paths above.
3. Allow inbound TCP `15443` in the hosting-provider firewall. If UFW is active,
   the script adds that rule automatically.
4. Use Debian 12 or later (Python 3.10+ is mandatory). The script installs the
   remaining OS and Python dependencies.

For the safest first run, choose your browser credentials before deploying:

```bash
export BACKTEST_BASIC_AUTH_USER='backtest_admin'
read -r -s -p 'Basic-auth password: ' BACKTEST_BASIC_AUTH_PASSWORD; echo
export BACKTEST_BASIC_AUTH_PASSWORD
```

The password must be at least 16 characters. If omitted on the first run, the
deployer creates a strong password and prints it once at the end. On later runs
it preserves the existing credential file unless a new password is supplied.

## Deploy

```bash
cd /home/debian/Simpler-Trader
sudo -E bash deploy/debian/deploy_backtest_web.sh
```

By default, the script provisions a local PostgreSQL database and writes its
generated credential to `configs/postgres.env` with restrictive permissions. It
then checks database access by opening the actual candle store, which also
creates the `candles` table/index when needed.

To use an already-managed PostgreSQL server instead, create the real,
non-template `configs/postgres.env` first (copy
`configs/postgres.env.example`, replace its password, and set its host), then:

```bash
cd /home/debian/Simpler-Trader
sudo -E BACKTEST_DATABASE_MODE=external bash deploy/debian/deploy_backtest_web.sh
```

The deployer refuses the example passwords `simpler_pass` and `postgres`.

## Optional variables

For a local database, these let you choose its database/role; otherwise strong
values are generated as needed:

```bash
export BACKTEST_DB_NAME=simpler_trader
export BACKTEST_DB_USER=simpler
export BACKTEST_DB_PASSWORD='a-long-random-database-password'
```

The panel fetches missing candles from Binance. If the server requires an
egress proxy, configure it on the first local-database deploy like this:

```bash
sudo -E BACKTEST_CANDLE_PROXY=http://proxy.example:8080 \
  bash deploy/debian/deploy_backtest_web.sh
```

For an external database, place `WEB_CANDLE_PROXY`,
`WEB_CANDLE_HTTP_PROXY`, or `WEB_CANDLE_HTTPS_PROXY` in `configs/postgres.env`.
The deployer warns if Binance cannot be reached; it does not block deployment
because a fully populated candle database can still run backtests. Do not use
`SKIP_BINANCE_CONNECTIVITY_CHECK=1` unless that is intentional.

## Verify and operate

```bash
sudo bash deploy/debian/deploy_backtest_web.sh --check
sudo systemctl status simpler-trader-backtest-web.service
sudo journalctl -u simpler-trader-backtest-web.service -f
sudo nginx -t
```

Open the EMA + AVWAP URL in a browser, authenticate, submit a short date range,
and wait for the WebSocket status to reach `COMPLETED`. The first run can take
longer because missing Binance candles are inserted into PostgreSQL.

Useful paths:

- Systemd unit: `/etc/systemd/system/simpler-trader-backtest-web.service`
- Runtime web settings: `/etc/simpler-trader/backtest-web.env`
- Nginx site: `/etc/nginx/sites-available/jzbe.jazebeh.ir-backtest.conf`
- Database config: `/home/debian/Simpler-Trader/configs/postgres.env`
- Persisted web jobs: `/home/debian/Simpler-Trader/results/web_backtests/`
- Nginx logs: `/var/log/nginx/jzbe.jazebeh.ir-backtest.*.log`

After a code update, rerun the deployer: it refreshes the virtual environment,
restarts the service, validates Nginx, and reloads it. The deployer also adds a
Certbot deploy hook that reloads Nginx after certificate renewal.
