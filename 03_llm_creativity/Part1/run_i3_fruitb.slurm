#!/bin/bash
#SBATCH --job-name=a3_i3_fruitb
#SBATCH --output=5350a3-i3-%j.out
#SBATCH --time=08:00:00
#SBATCH --mem=12G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

set -euo pipefail

# 1) Modules first
module --force purge
module load ollama/0.9.3-GCCcore-13.3.0-CUDA-12.6.0

# 2) Then conda
source ~/.bashrc
conda activate 5350

cd "$HOME/5350 Assignment3/Assignment3/Part1"

echo "Python executable:"
which python
python -c "import sys; print(sys.executable)"
python -c "import tqdm; print('tqdm ok', tqdm.__version__)"

export OLLAMA_PORT=11434

# 3) Start Ollama server
ollama serve > ollama_server.log 2>&1 &
sleep 2

for i in {1..90}; do
  if curl -s "http://127.0.0.1:${OLLAMA_PORT}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is up"
    break
  fi
  sleep 1
done

# 4) Pull model only if missing
ollama list | grep -q "qwen2.5:7b" || ollama pull qwen2.5:7b

# 5) Ensure prompt files for I.3 exist
mkdir -p prompts/bfruit

cat > prompts/bfruit/plain.txt <<'EOF'
Return exactly one fruit that starts with the letter b.
Output only the fruit name in lowercase, with no punctuation and no explanation.
EOF

cat > prompts/bfruit/scattergories.txt <<'EOF'
You are playing Scattergories.
Letter: b
Category: fruits
Return exactly one valid Scattergories answer that starts with b.
Try to avoid repeatedly choosing the most common answer across runs, while keeping the answer realistic and specific.
Output only lowercase letters and spaces.
EOF

cat > prompts/bfruit/strict_validation.txt <<'EOF'
Task: output exactly one real fruit name that starts with b.
Validity rules:
1. It must be an edible fruit.
2. It must start with the letter b.
3. No brand names, no dishes, no vegetables.
Output format: one lowercase phrase only, no punctuation, no explanation.
EOF

cat > prompts/bfruit/diversity_push.txt <<'EOF'
Return exactly one fruit that starts with b.
Avoid always choosing the most common answer.
Still ensure the answer is a real fruit and starts with b.
Output only lowercase letters and spaces, no punctuation, no explanation.
EOF

cat > prompts/bfruit/enumerate_then_pick.txt <<'EOF'
Before answering, silently recall at least 8 different real fruits that start with b — ranging from common to uncommon varieties. Do not output this list.

Then assign each recalled fruit a number starting from 1.
Pick a number at random from 1 to your count, trying to avoid always picking the same position.
Output only the fruit at that number.

Rules:
- Must be a real, edible fruit.
- Must start with the letter b.
- No brand names, no vegetables, no dishes.
Output format: one lowercase word or phrase, no punctuation, no explanation.
EOF

PY="$CONDA_PREFIX/bin/python"
MODEL="qwen2.5:7b"
SAMPLES=500
TOPK=40

run_calibration () {
  local prompt_file="$1"
  local prompt_id="$2"
  local outdir="outputs/fruitb/${prompt_id}"
  echo "Running prompt_id=${prompt_id}"

  "$PY" assignment3_starter.py calibrate \
    --model "$MODEL" \
    --task fruitb \
    --samples "$SAMPLES" \
    --top-k "$TOPK" \
    --prompt-file "$prompt_file" \
    --prompt-id "$prompt_id" \
    --outdir "$outdir"
}

# I.3 runs: includes Scattergories vs non-Scattergories comparison
run_calibration "prompts/bfruit/plain.txt" "plain"
run_calibration "prompts/bfruit/scattergories.txt" "scattergories"
run_calibration "prompts/bfruit/strict_validation.txt" "strict_validation"
run_calibration "prompts/bfruit/diversity_push.txt" "diversity_push"
run_calibration "prompts/bfruit/enumerate_then_pick.txt" "enumerate_then_pick"

echo "All I.3 calibrations finished."
echo "Outputs:"
echo "  outputs/fruitb/plain"
echo "  outputs/fruitb/scattergories"
echo "  outputs/fruitb/strict_validation"
echo "  outputs/fruitb/diversity_push"
echo "  outputs/fruitb/enumerate_then_pick"
