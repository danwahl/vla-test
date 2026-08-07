source /opt/venv/openpi/bin/activate

export RLINF_DIR=/opt/rlinf
export VLA_TEST_DIR=/root/vla-test
export EMBODIED_PATH=$RLINF_DIR/examples/embodiment
export PYTHONPATH=$RLINF_DIR:$VLA_TEST_DIR/sim/src
export SFT_CKPT=/data/checkpoints/pi05_so101_block_stack_sim/openpi
export RL_LAYOUTS=$VLA_TEST_DIR/rl_layouts.jsonl
