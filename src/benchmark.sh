set -e

cd "$(dirname "$0")"

python3 benchmark.py

cd -
