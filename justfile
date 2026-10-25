install:
    scripts/dev-setup.sh

lint:
    uv run ruff check bench

profile *args:
    uv run python bench/randprof.py weights/corridorkey_v2.safetensors {{args}}

bench *args:
    uv run python bench/fps.py {{args}}
