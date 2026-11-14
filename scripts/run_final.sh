#!/usr/bin/env bash
# Run every final experiment, then analyse each one.
#
#   bash scripts/run_final.sh                 # full runs (~1 hour on 4 cores)
#   QUICK=1 bash scripts/run_final.sh         # 2 populations each, to check it all works
#   WORKERS=8 bash scripts/run_final.sh       # more CPU cores
#   ONLY="main types" bash scripts/run_final.sh   # just some experiments
#
# Results:  results/final/<experiment>/<timestamp>/   (raw; git-ignored)
# Analysis: reports/final/<experiment>/               (tables + figures; commit these)
set -euo pipefail
cd "$(dirname "$0")/.."

WORKERS=${WORKERS:-4}
NPOP_ARG=()
if [[ "${QUICK:-0}" == "1" ]]; then NPOP_ARG=(--n-pop 2); fi

if [[ "${QUICK:-0}" == "1" ]]; then
  python scripts/make_final_configs.py --n-pop 2
else
  python scripts/make_final_configs.py
fi

ALL="none random popularity content bm25 mf calibrated fatigue_popularity fatigue_mf fatigue_content ideal mmr"
CORE="none random popularity content bm25 mf mmr ideal"
MMR_SWEEP=(--mmr-diversity-weights 0 0.3 0.6 0.9 1.0)
MMR_ONE=(--mmr-diversity-weight 0.6)

want() { [[ -z "${ONLY:-}" ]] || [[ " ${ONLY} " == *" $1 "* ]]; }

run() {   # run <name> <config> [extra args...]
  local name=$1 cfg=$2; shift 2
  echo "=== $name ==="
  python scripts/run_news.py --config "$cfg" --out "results/final/$name" \
      --workers "$WORKERS" --no-archive "${NPOP_ARG[@]}" "$@"
  local latest
  latest=$(ls -td "results/final/$name"/*/ | head -1)
  python scripts/analyse_results.py "$latest" --out "reports/final/$name" "${MAIN[@]}" >/dev/null
  echo "    analysis -> reports/final/$name"
}

# 1. Main comparison: all recommenders, MMR sweep, fatigue strengths
MAIN=(--main rho=0.5 phi=0.5 K=10)
want main && run main configs/final/main.yaml --recommenders $ALL "${MMR_SWEEP[@]}"

# 2. Reader types (cautious / explorer / easily bored / average)
MAIN=(--main rho=0.5 K=10)
want types && run types configs/final/types.yaml --recommenders $ALL "${MMR_ONE[@]}"

# 3. Slate size: how much of "no recommender"'s advantage is choice-set size
MAIN=(--main rho=0.5 phi=0.5 K=10)
want slate && run slate configs/final/main.yaml --recommenders $CORE "${MMR_ONE[@]}" \
    --slate-sizes 5 10 25 50 500

# 4. Returning users: recommenders see readers' earlier history
MAIN=(--main rho=0.5 phi=0.5 K=10 reader_history_visible=False)
want returning && run returning configs/final/main.yaml \
    --recommenders none popularity content bm25 mf calibrated mmr "${MMR_ONE[@]}" \
    --reader-history-visible both

# 5. Session dropout
MAIN=(--main rho=0.5 phi=1.0 outside_option=-1.0 K=10)
want dropout && run dropout configs/final/dropout.yaml --recommenders $CORE "${MMR_ONE[@]}"

# 6-9. Sensitivity and robustness
MAIN=(--main rho=0.5 phi=0.5 uncertainty_n0=20 K=10)
want n0 && run n0 configs/final/n0.yaml --recommenders $CORE "${MMR_ONE[@]}"
MAIN=(--main rho=0.5 phi=0.5 K=10)
want rho && run rho configs/final/rho.yaml --recommenders $CORE "${MMR_ONE[@]}"
MAIN=(--main gamma=1.0 beta=1.0 sigma_bar=1.0 K=10)
want robustness && run robustness configs/final/robustness.yaml --recommenders $CORE "${MMR_ONE[@]}"
MAIN=(--main rho=0.5 phi=0.5 K=10)
want pca4 && run pca4 configs/final/pca4.yaml --recommenders $CORE "${MMR_ONE[@]}"
want pca16 && run pca16 configs/final/pca16.yaml --recommenders $CORE "${MMR_ONE[@]}"

# 10. Validation on real MIND data (no simulation)
if want validation; then
  echo "=== validation ==="
  python scripts/validate_reader_choices.py --out reports/final/validation
  python scripts/estimate_fatigue.py --out-dir reports/final/validation
fi

echo "All done. Commit reports/final/ (not results/)."
