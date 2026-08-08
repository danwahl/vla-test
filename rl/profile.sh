# Where a run reads its pieces from, sourced by the image's profile script once
# RLINF_DIR and VLA_TEST_DIR are set.

export EMBODIED_PATH=$RLINF_DIR/examples/embodiment
export PYTHONPATH=$RLINF_DIR:$VLA_TEST_DIR/sim/src
export SFT_CKPT=/data/checkpoints/pi05_so101_block_stack_sim/openpi
export RL_LAYOUTS=$VLA_TEST_DIR/rl_layouts.jsonl
export EVAL_LAYOUTS=$VLA_TEST_DIR/eval_layouts.jsonl
