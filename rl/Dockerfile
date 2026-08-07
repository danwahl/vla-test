# The environment a run happens in, built from the repository root:
#
#     docker build -f rl/Dockerfile -t danwahl/vla-test-rl .
#
# It contains the base RLinf code. This repository is copied onto a machine running it and
# patched in there, so a config change needs only an rsync.
#
# RLinf's image has a venv per embodiment. The openpi one includes ManiSkill, openpi, a CUDA
# torch and the packages RLinf imports.
FROM rlinf/rlinf:agentic-rlinf0.4-maniskill_libero

RUN apt-get update -qq \
    && apt-get install -y -qq --no-install-recommends openssh-server rsync \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /run/sshd \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config

# The commit rl/patch.py is written against. Each of its edits asserts a single anchor
# match, so a move here without a matching patch update fails at the patch.
ARG RLINF_COMMIT=d3aff547c06d66e2e792a62a71122abed86e8aee
RUN git clone --filter=blob:none https://github.com/RLinf/RLinf.git /opt/rlinf \
    && git -C /opt/rlinf checkout --detach "$RLINF_COMMIT" \
    && rm -rf /opt/rlinf/.git

# An ssh session starts with an empty environment, so the settings a run reads are in a
# profile script, and commands run in a login shell.
COPY rl/profile.sh /etc/profile.d/vla-test.sh
# A login shell reads /root/.bashrc after /etc/profile.d, and the image activates its
# openvla venv there, so the openpi one is selected again at the end.
RUN echo 'source /etc/profile.d/vla-test.sh' >> /root/.bashrc
SHELL ["/bin/bash", "-lc"]

# RunPod sets PUBLIC_KEY in the container.
CMD mkdir -p /root/.ssh \
    && echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys \
    && chmod 700 /root/.ssh \
    && chmod 600 /root/.ssh/authorized_keys \
    && exec /usr/sbin/sshd -D -e
