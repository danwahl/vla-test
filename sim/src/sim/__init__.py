"""The SO-101 in simulation.

`spec` describes the arm, its cameras and the task, and imports nothing that simulates
them, so the hardware package reaches it on a machine with no physics engine installed.
Importing `sim.env` is what registers the ``so101`` agent and ``SO101BlockStack-v1``.
"""
