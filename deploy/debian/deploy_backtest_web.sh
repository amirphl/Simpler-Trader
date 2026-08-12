#!/usr/bin/env bash
# Deploy the Simpler-Trader backtest panel on Debian behind existing Nginx.
#
# Default target:
#   https://jzbe.jazebeh.ir:15443/static/ema_avwap_pullback.html
#
# Run from the checked-out project on the server:
#   sudo bash deploy/debian/deploy_backtest_web.sh
#
# Set BACKTEST_DATABASE_MODE=external when configs/postgres.env already points
# at an external PostgreSQL service. See deploy/debian/README.md for variables.

set -Eeuo pipefail
IFS=$'\n\t'
umask 077

readonly APP_USER="debian"
readonly APP_GROUP="debian"
readonly NGINX_GROUP="www-data"
readonly PROJECT_ROOT="/home/debian/Simpler-Trader"
readonly DOMAIN="jzbe.jazebeh.ir"
readonly HTTPS_PORT="15443"
readonly BACKEND_HOST="127.0.0.1"
readonly BACKEND_PORT="9092"
readonly DATABASE_MODE="${BACKTEST_DATABASE_MODE:-local}"
readonly VENV_DIR="${PROJECT_ROOT}/.venv"
readonly PYTHON_BIN="${VENV_DIR}/bin/python"
readonly POSTGRES_ENV="${PROJECT_ROOT}/configs/postgres.env"
readonly SERVICE_NAME="simpler-trader-backtest-web.service"
readonly SERVICE_PATH="/etc/systemd/system/${SERVICE_NAME}"
readonly NGINX_AVAILABLE="/etc/nginx/sites-available/${DOMAIN}-backtest.conf"
readonly NGINX_ENABLED="/etc/nginx/sites-enabled/${DOMAIN}-backtest.conf"
readonly HTPASSWD_FILE="/etc/nginx/.htpasswd-simpler-trader-backtest"
readonly RUNTIME_ENV_DIR="/etc/simpler-trader"
readonly RUNTIME_ENV="${RUNTIME_ENV_DIR}/backtest-web.env"
readonly NFTABLES_MAIN_CONFIG="/etc/nftables.conf"
readonly NFTABLES_DROPIN_DIR="/etc/nftables.d"
readonly NFTABLES_DROPIN="${NFTABLES_DROPIN_DIR}/simpler-trader-backtest.nft"
readonly NFTABLES_DROPIN_INCLUDE="include \"${NFTABLES_DROPIN_DIR}/*.nft\""
readonly CERT_DIR="/etc/letsencrypt/live/${DOMAIN}"
readonly CERT_FILE="${CERT_DIR}/fullchain.pem"
readonly KEY_FILE="${CERT_DIR}/privkey.pem"
readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly CHECK_ONLY="${1:-}"

GENERATED_BASIC_AUTH_PASSWORD=""

log() {
    printf '[deploy] %s\n' "$*"
}

die() {
    printf '[deploy] ERROR: %s\n' "$*" >&2
    exit 1
}

warn() {
    printf '[deploy] WARNING: %s\n' "$*" >&2
}

on_error() {
    local exit_code=$?
    printf '[deploy] ERROR: command failed (exit %s) at line %s.\n' "$exit_code" "$1" >&2
    exit "$exit_code"
}
trap 'on_error "$LINENO"' ERR

usage() {
    cat <<'EOF'
Usage: sudo bash deploy/debian/deploy_backtest_web.sh [--check]

Without arguments this installs prerequisites and deploys the service.
--check only validates the host and required pre-existing files; it makes no changes.

Optional environment variables:
  BACKTEST_DATABASE_MODE=local|external  (default: local)
  BACKTEST_DB_NAME, BACKTEST_DB_USER, BACKTEST_DB_PASSWORD
  BACKTEST_BASIC_AUTH_USER, BACKTEST_BASIC_AUTH_PASSWORD
  BACKTEST_CANDLE_PROXY                (for Binance access; written only when config is created)
  SKIP_BINANCE_CONNECTIVITY_CHECK=1
EOF
}

require_root() {
    [[ "$(id -u)" -eq 0 ]] || die "Run this script with sudo or as root."
}

validate_static_settings() {
    [[ "$DATABASE_MODE" == "local" || "$DATABASE_MODE" == "external" ]] \
        || die "BACKTEST_DATABASE_MODE must be local or external."
}

check_os() {
    [[ -r /etc/os-release ]] || die "Cannot determine the operating system."
    # shellcheck disable=SC1091
    . /etc/os-release
    [[ "${ID:-}" == "debian" ]] || die "This script supports Debian only (detected: ${ID:-unknown})."
    command -v systemctl >/dev/null || die "systemd is required."
    [[ -d /run/systemd/system ]] || die "The host is not running systemd."
}

check_project() {
    [[ -d "$PROJECT_ROOT" ]] || die "Project root does not exist: $PROJECT_ROOT"
    [[ -f "$PROJECT_ROOT/requirements.txt" ]] || die "requirements.txt is missing from $PROJECT_ROOT"
    [[ -f "$PROJECT_ROOT/cmd/web/main.py" ]] || die "The backtest web module is missing from $PROJECT_ROOT"
    id "$APP_USER" >/dev/null 2>&1 || die "Application user does not exist: $APP_USER"
    getent group "$APP_GROUP" >/dev/null || die "Application group does not exist: $APP_GROUP"
    getent group "$NGINX_GROUP" >/dev/null || die "Nginx group does not exist: $NGINX_GROUP"
}

check_certificates() {
    [[ -r "$CERT_FILE" ]] || die "TLS certificate is missing or unreadable: $CERT_FILE"
    [[ -r "$KEY_FILE" ]] || die "TLS private key is missing or unreadable: $KEY_FILE"
    command -v openssl >/dev/null || die "openssl is required."
    openssl x509 -in "$CERT_FILE" -noout >/dev/null
    openssl pkey -in "$KEY_FILE" -noout >/dev/null
    if ! openssl x509 -checkend 86400 -noout -in "$CERT_FILE" >/dev/null; then
        warn "The certificate expires within 24 hours; renew it before deployment."
    fi
}

check_python_version() {
    command -v python3 >/dev/null || die "python3 is not installed."
    python3 - <<'PY'
import sys
if sys.version_info < (3, 10):
    raise SystemExit(
        f"Python 3.10+ is required; found {sys.version.split()[0]}. "
        "Use Debian 12+ or install a supported Python before continuing."
    )
PY
}

check_preexisting_database_config() {
    if [[ "$DATABASE_MODE" == "external" && ! -s "$POSTGRES_ENV" ]]; then
        die "External database mode requires a populated $POSTGRES_ENV."
    fi
    if [[ -s "$POSTGRES_ENV" ]]; then
        if grep -Eq '^CANDLE_DB_PASSWORD=(simpler_pass|postgres)$' "$POSTGRES_ENV"; then
            die "Refusing to deploy with the example PostgreSQL password. Set a strong database password first."
        fi
    fi
}

check_port_conflict() {
    command -v ss >/dev/null || return 0
    if ss -ltn "sport = :${HTTPS_PORT}" | grep -Eq ":${HTTPS_PORT}[[:space:]]"; then
        if [[ ! -f "$NGINX_AVAILABLE" ]] || ! grep -Fq "server_name ${DOMAIN};" "$NGINX_AVAILABLE"; then
            die "TCP port ${HTTPS_PORT} is already in use. Resolve that listener before deploying."
        fi
    fi
}

install_packages() {
    local -a packages=(ca-certificates curl openssl python3 python3-venv python3-pip nginx apache2-utils iproute2 nftables)
    if [[ "$DATABASE_MODE" == "local" ]]; then
        packages+=(postgresql postgresql-client)
    fi
    log "Installing required Debian packages."
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"
}

run_as_app() {
    runuser --user "$APP_USER" -- "$@"
}

run_as_postgres() {
    # runuser preserves the caller's working directory. The application
    # checkout is deliberately not traversable by the postgres account.
    runuser --user postgres -- sh -c 'cd / && exec "$@"' sh "$@"
}

prepare_virtualenv() {
    install -d -o "$APP_USER" -g "$APP_GROUP" -m 0750 \
        "$PROJECT_ROOT/data" "$PROJECT_ROOT/results" "$PROJECT_ROOT/logs"
    if [[ ! -x "$PYTHON_BIN" ]]; then
        log "Creating Python virtual environment."
        run_as_app python3 -m venv "$VENV_DIR"
    else
        # .venv is a generated deployment artifact; make an interrupted prior
        # root-run install usable by the unprivileged service account again.
        chown -R "$APP_USER:$APP_GROUP" "$VENV_DIR"
    fi
    log "Installing Python dependencies into $VENV_DIR."
    run_as_app "$PYTHON_BIN" -m pip install --upgrade pip
    run_as_app "$PYTHON_BIN" -m pip install -r "$PROJECT_ROOT/requirements.txt"
    runuser --user "$APP_USER" -- sh -c 'cd "$1" && exec "$2" -c "from fastapi import FastAPI; from webserver.app import app; assert app.title"' \
        sh "$PROJECT_ROOT" "$PYTHON_BIN"
}

validate_identifier() {
    local value="$1"
    local label="$2"
    [[ "$value" =~ ^[A-Za-z_][A-Za-z0-9_]{0,62}$ ]] || die "$label must be a PostgreSQL identifier."
}

sql_literal() {
    local value="$1"
    [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] \
        || die "Database values must not contain newline characters."
    value=${value//\'/\'\'}
    printf "'%s'" "$value"
}

provision_local_database() {
    local db_name="${BACKTEST_DB_NAME:-simpler_trader}"
    local db_user="${BACKTEST_DB_USER:-simpler}"
    local db_password="${BACKTEST_DB_PASSWORD:-}"
    validate_identifier "$db_name" "BACKTEST_DB_NAME"
    validate_identifier "$db_user" "BACKTEST_DB_USER"
    if [[ -z "$db_password" ]]; then
        db_password="$(openssl rand -hex 32)"
    fi
    [[ ${#db_password} -ge 20 ]] || die "BACKTEST_DB_PASSWORD must be at least 20 characters."

    systemctl enable --now postgresql
    log "Provisioning local PostgreSQL database $db_name."
    if ! run_as_postgres psql --tuples-only --no-align \
        --command "SELECT 1 FROM pg_roles WHERE rolname = $(sql_literal "$db_user")" | grep -qx '1'; then
        run_as_postgres psql --set=ON_ERROR_STOP=1 \
            --command "CREATE ROLE \"${db_user}\" LOGIN PASSWORD $(sql_literal "$db_password")"
    else
        # Keep a supplied/generated password in sync on repeat deployments.
        run_as_postgres psql --set=ON_ERROR_STOP=1 \
            --command "ALTER ROLE \"${db_user}\" PASSWORD $(sql_literal "$db_password")"
    fi
    if ! run_as_postgres psql --tuples-only --no-align \
        --command "SELECT 1 FROM pg_database WHERE datname = $(sql_literal "$db_name")" | grep -qx '1'; then
        run_as_postgres createdb --owner="$db_user" "$db_name"
    fi

    install -d -o "$APP_USER" -g "$APP_GROUP" -m 0750 "$(dirname "$POSTGRES_ENV")"
    install -o "$APP_USER" -g "$APP_GROUP" -m 0640 /dev/null "$POSTGRES_ENV"
    cat >"$POSTGRES_ENV" <<EOF
CANDLE_DB_HOST=127.0.0.1
CANDLE_DB_PORT=5432
CANDLE_DB_USER=${db_user}
CANDLE_DB_PASSWORD=${db_password}
CANDLE_DB_NAME=${db_name}
CANDLE_DB_SSLMODE=disable
CANDLE_DB_MIN_POOL_SIZE=1
CANDLE_DB_MAX_POOL_SIZE=8
CANDLE_DB_CONNECT_TIMEOUT=10
CANDLE_DB_MAX_IDLE_SECONDS=300
EOF
    if [[ -n "${BACKTEST_CANDLE_PROXY:-}" ]]; then
        printf 'WEB_CANDLE_PROXY=%s\n' "$BACKTEST_CANDLE_PROXY" >>"$POSTGRES_ENV"
    fi
    chown "$APP_USER:$APP_GROUP" "$POSTGRES_ENV"
    chmod 0640 "$POSTGRES_ENV"
}

ensure_database_config() {
    if [[ ! -s "$POSTGRES_ENV" ]]; then
        [[ "$DATABASE_MODE" == "local" ]] || die "Missing PostgreSQL config: $POSTGRES_ENV"
        provision_local_database
    else
        chown "$APP_USER:$APP_GROUP" "$POSTGRES_ENV"
        chmod 0640 "$POSTGRES_ENV"
    fi
}

verify_database() {
    log "Checking PostgreSQL connectivity and candle-schema access."
    runuser --user "$APP_USER" -- sh -c 'cd "$1" && export CANDLE_DB_ENV_FILE="$2" && exec "$3" -c "from candle_downloader.storage import build_store; store = build_store(\"postgres\"); store.close()"' \
        sh "$PROJECT_ROOT" "$POSTGRES_ENV" "$PYTHON_BIN"
}

verify_database_connection_only() {
    log "Checking PostgreSQL connectivity."
    runuser --user "$APP_USER" -- sh -c 'cd "$1" && export CANDLE_DB_ENV_FILE="$2" && exec "$3" -c "import os, psycopg; from pathlib import Path; from candle_downloader.storage import PostgresConfig; connection = psycopg.connect(PostgresConfig.from_env(Path(os.environ[\"CANDLE_DB_ENV_FILE\"])).to_conninfo()); connection.close()"' \
        sh "$PROJECT_ROOT" "$POSTGRES_ENV" "$PYTHON_BIN"
}

write_runtime_environment() {
    install -d -o root -g "$APP_GROUP" -m 0750 "$RUNTIME_ENV_DIR"
    install -o root -g "$APP_GROUP" -m 0640 /dev/null "$RUNTIME_ENV"
    cat >"$RUNTIME_ENV" <<EOF
WEB_LOG_LEVEL=info
WEB_FORCE_HTTPS=true
WEB_TRUSTED_HOSTS=${DOMAIN},${DOMAIN}:${HTTPS_PORT},localhost,127.0.0.1
WEB_ALLOWED_ORIGINS=https://${DOMAIN}:${HTTPS_PORT}
PYTHONUNBUFFERED=1
EOF
}

configure_basic_auth() {
    local auth_user="${BACKTEST_BASIC_AUTH_USER:-backtest_admin}"
    local auth_password="${BACKTEST_BASIC_AUTH_PASSWORD:-}"
    [[ "$auth_user" =~ ^[A-Za-z0-9._-]+$ ]] || die "BACKTEST_BASIC_AUTH_USER contains unsupported characters."

    if [[ -z "$auth_password" && -f "$HTPASSWD_FILE" ]]; then
        log "Keeping existing Nginx basic-auth credentials."
        chown "root:${NGINX_GROUP}" "$HTPASSWD_FILE"
        chmod 0640 "$HTPASSWD_FILE"
        return
    fi
    if [[ -z "$auth_password" ]]; then
        auth_password="$(openssl rand -base64 24)"
        GENERATED_BASIC_AUTH_PASSWORD="$auth_password"
    fi
    [[ ${#auth_password} -ge 16 ]] || die "BACKTEST_BASIC_AUTH_PASSWORD must be at least 16 characters."
    printf '%s\n' "$auth_password" | htpasswd -iB -c "$HTPASSWD_FILE" "$auth_user" >/dev/null
    # Authentication is evaluated by unprivileged Nginx worker processes.
    # Keep the file root-owned while granting only Nginx's service group read access.
    chown "root:${NGINX_GROUP}" "$HTPASSWD_FILE"
    chmod 0640 "$HTPASSWD_FILE"
}

install_service_and_nginx() {
    install -m 0644 "$SCRIPT_DIR/backtest-web-jzbe.service" "$SERVICE_PATH"
    install -m 0644 "$SCRIPT_DIR/jzbe.jazebeh.ir-backtest.conf" "$NGINX_AVAILABLE"
    ln -sfn "$NGINX_AVAILABLE" "$NGINX_ENABLED"

    systemctl daemon-reload
    systemctl enable "$SERVICE_NAME"
    systemctl restart "$SERVICE_NAME"
    if ! systemctl is-active --quiet "$SERVICE_NAME"; then
        journalctl --no-pager -u "$SERVICE_NAME" -n 80 >&2 || true
        die "The backtest service did not start."
    fi
    wait_for_backtest_backend

    nginx -t
    nginx -T 2>&1 | grep -Fq "configuration file ${NGINX_ENABLED}:" \
        || die "Nginx is not including ${NGINX_ENABLED}. Add Debian's sites-enabled include before retrying."
    systemctl enable nginx
    systemctl reload-or-restart nginx
    systemctl is-active --quiet nginx || die "Nginx did not start."
    curl --silent --show-error --max-time 15 --resolve \
        "${DOMAIN}:${HTTPS_PORT}:127.0.0.1" \
        -o /dev/null -w '%{http_code}' "https://${DOMAIN}:${HTTPS_PORT}/" | grep -qx '401'
}

wait_for_backtest_backend() {
    local attempt
    for ((attempt = 1; attempt <= 30; attempt++)); do
        if curl --fail --silent --show-error --max-time 2 \
            -H 'X-Forwarded-Proto: https' "http://${BACKEND_HOST}:${BACKEND_PORT}/" >/dev/null; then
            log "Backtest service is ready on ${BACKEND_HOST}:${BACKEND_PORT}."
            return
        fi
        if ! systemctl is-active --quiet "$SERVICE_NAME"; then
            journalctl --no-pager -u "$SERVICE_NAME" -n 80 >&2 || true
            die "The backtest service stopped before it became ready."
        fi
        sleep 1
    done
    journalctl --no-pager -u "$SERVICE_NAME" -n 80 >&2 || true
    die "The backtest service did not listen on ${BACKEND_HOST}:${BACKEND_PORT} within 30 seconds."
}

configure_certificate_reload_hook() {
    local hook_dir="/etc/letsencrypt/renewal-hooks/deploy"
    local hook_path="${hook_dir}/reload-nginx-simpler-trader-backtest"
    install -d -o root -g root -m 0755 "$hook_dir"
    cat >"$hook_path" <<'EOF'
#!/usr/bin/env sh
systemctl reload nginx
EOF
    chown root:root "$hook_path"
    chmod 0755 "$hook_path"
}

check_binance_connectivity() {
    [[ "${SKIP_BINANCE_CONNECTIVITY_CHECK:-0}" == "1" ]] && return
    local -a curl_args=(--fail --silent --show-error --connect-timeout 10 --max-time 20)
    local candle_proxy="${BACKTEST_CANDLE_PROXY:-}"
    if [[ -z "$candle_proxy" && -s "$POSTGRES_ENV" ]]; then
        candle_proxy="$(sed -nE 's/^(WEB_CANDLE_HTTPS_PROXY|WEB_CANDLE_PROXY)=//p' "$POSTGRES_ENV" | tail -n 1)"
    fi
    if [[ -n "$candle_proxy" ]]; then
        curl_args+=(--proxy "$candle_proxy")
    fi
    if ! curl "${curl_args[@]}" 'https://api.binance.com/api/v3/ping' >/dev/null; then
        warn "Binance is unreachable from this host. Existing candles still work, but downloads will fail. Set BACKTEST_CANDLE_PROXY and rerun, or use SKIP_BINANCE_CONNECTIVITY_CHECK=1 only when the database is already complete."
    fi
}

ensure_local_postgres_is_loopback_only() {
    [[ "$DATABASE_MODE" == "local" ]] || return
    local listeners
    listeners="$(ss -ltn "sport = :5432" | awk 'NR > 1 { print $4 }')"
    [[ -n "$listeners" ]] || die "PostgreSQL is not listening on TCP port 5432."
    if printf '%s\n' "$listeners" | grep -Evq '^(127\.0\.0\.1|\[::1\]):5432$'; then
        die "PostgreSQL has a non-loopback listener (${listeners}). Set listen_addresses to localhost/127.0.0.1 before continuing."
    fi
}

check_nftables_firewall() {
    command -v nft >/dev/null || die "nftables is required."
    [[ -f "$NFTABLES_MAIN_CONFIG" ]] || die "nftables config is missing: $NFTABLES_MAIN_CONFIG"
    [[ -f "$NFTABLES_DROPIN" ]] || die "nftables backtest rule file is missing: $NFTABLES_DROPIN"
    grep -Fq "$NFTABLES_DROPIN_INCLUDE" "$NFTABLES_MAIN_CONFIG" \
        || die "${NFTABLES_MAIN_CONFIG} does not include ${NFTABLES_DROPIN_DIR}."
    nft list chain inet filter input | grep -Fq 'tcp dport 5432 drop' \
        || die "Missing nftables inbound PostgreSQL deny rule."
    nft list chain inet filter input | grep -Fq "tcp dport ${HTTPS_PORT} accept" \
        || die "Missing nftables TCP ${HTTPS_PORT} allow rule."
}

configure_nftables_firewall() {
    command -v nft >/dev/null || die "nftables is required."
    [[ -f "$NFTABLES_MAIN_CONFIG" ]] \
        || die "Expected nftables configuration is missing: $NFTABLES_MAIN_CONFIG"
    grep -Fq 'table inet filter' "$NFTABLES_MAIN_CONFIG" \
        || die "Expected table inet filter is not defined in $NFTABLES_MAIN_CONFIG"
    grep -Eq '^[[:space:]]*chain[[:space:]]+input[[:space:]]*\{' "$NFTABLES_MAIN_CONFIG" \
        || die "Expected input chain is not defined in $NFTABLES_MAIN_CONFIG"

    install -d -o root -g root -m 0755 "$NFTABLES_DROPIN_DIR"
    install -o root -g root -m 0644 /dev/null "$NFTABLES_DROPIN"
    cat >"$NFTABLES_DROPIN" <<EOF
# Managed by Simpler-Trader's Debian deployment.
# PostgreSQL remains local-only; deny the port explicitly even though the
# input chain's policy is drop, so later broad ACCEPT rules cannot expose it.
add rule inet filter input tcp dport 5432 drop comment "simpler-trader-backtest: deny PostgreSQL"
add rule inet filter input tcp dport ${HTTPS_PORT} accept comment "simpler-trader-backtest: allow HTTPS"
EOF

    if ! grep -Fq "$NFTABLES_DROPIN_INCLUDE" "$NFTABLES_MAIN_CONFIG"; then
        printf '\n%s\n' "$NFTABLES_DROPIN_INCLUDE" >>"$NFTABLES_MAIN_CONFIG"
    fi

    # Check the complete persisted ruleset before replacing the active one.
    nft --check --file "$NFTABLES_MAIN_CONFIG"
    systemctl enable nftables
    systemctl reload-or-restart nftables
    systemctl is-active --quiet nftables || die "nftables did not start."
    check_nftables_firewall
    log "nftables configured: TCP ${HTTPS_PORT} allowed; inbound PostgreSQL TCP 5432 denied."
}

check_running_deployment() {
    [[ -f "$SERVICE_PATH" ]] || die "Systemd unit is missing: $SERVICE_PATH"
    [[ -f "$RUNTIME_ENV" ]] || die "Runtime web environment is missing: $RUNTIME_ENV"
    [[ -f "$NGINX_AVAILABLE" && -e "$NGINX_ENABLED" ]] \
        || die "Nginx backtest site is not enabled."
    [[ -f "$HTPASSWD_FILE" ]] || die "Nginx basic-auth file is missing: $HTPASSWD_FILE"
    systemctl is-active --quiet "$SERVICE_NAME" \
        || die "Backtest service is not active: $SERVICE_NAME"
    curl --fail --silent --show-error --max-time 15 \
        -H 'X-Forwarded-Proto: https' "http://${BACKEND_HOST}:${BACKEND_PORT}/" >/dev/null
    nginx -t
    nginx -T 2>&1 | grep -Fq "configuration file ${NGINX_ENABLED}:" \
        || die "Nginx is not including ${NGINX_ENABLED}."
    systemctl is-active --quiet nginx || die "Nginx is not active."
    curl --silent --show-error --max-time 15 --resolve \
        "${DOMAIN}:${HTTPS_PORT}:127.0.0.1" \
        -o /dev/null -w '%{http_code}' "https://${DOMAIN}:${HTTPS_PORT}/" | grep -qx '401' \
        || die "The public TLS listener did not return the expected HTTP 401 basic-auth challenge."
}

run_check_only() {
    check_os
    check_project
    check_certificates
    check_python_version
    check_preexisting_database_config
    check_port_conflict
    command -v nginx >/dev/null || die "nginx is not installed. Run without --check to install it."
    command -v curl >/dev/null || die "curl is not installed. Run without --check to install it."
    command -v htpasswd >/dev/null || die "apache2-utils (htpasswd) is not installed. Run without --check to install it."
    [[ -x "$PYTHON_BIN" ]] || die "Virtual environment is missing: $VENV_DIR. Run without --check to create it."
    [[ -s "$POSTGRES_ENV" ]] || die "PostgreSQL config is missing: $POSTGRES_ENV. Run without --check to create it."
    ensure_local_postgres_is_loopback_only
    verify_database_connection_only
    check_nftables_firewall
    check_running_deployment
    log "All non-mutating deployment checks passed."
}

main() {
    if [[ "$CHECK_ONLY" == "--help" || "$CHECK_ONLY" == "-h" ]]; then
        usage
        return
    fi
    [[ -z "$CHECK_ONLY" || "$CHECK_ONLY" == "--check" ]] || { usage >&2; exit 2; }

    require_root
    validate_static_settings
    if [[ "$CHECK_ONLY" == "--check" ]]; then
        run_check_only
        return
    fi

    check_os
    check_project
    install_packages
    check_python_version
    check_certificates
    check_preexisting_database_config
    check_port_conflict
    if [[ "$DATABASE_MODE" == "local" ]]; then
        systemctl enable --now postgresql
        ensure_local_postgres_is_loopback_only
    fi
    prepare_virtualenv
    ensure_database_config
    verify_database
    write_runtime_environment
    configure_basic_auth
    configure_nftables_firewall
    install_service_and_nginx
    configure_certificate_reload_hook
    check_binance_connectivity

    log "Deployment complete."
    log "EMA + AVWAP panel: https://${DOMAIN}:${HTTPS_PORT}/static/ema_avwap_pullback.html"
    if [[ -n "$GENERATED_BASIC_AUTH_PASSWORD" ]]; then
        log "Generated basic-auth user: ${BACKTEST_BASIC_AUTH_USER:-backtest_admin}"
        log "Generated basic-auth password (save it now): ${GENERATED_BASIC_AUTH_PASSWORD}"
    fi
    log "Check status: systemctl status ${SERVICE_NAME}"
}

main "$@"
