FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-devel AS base


# Fix python for SSH sessions
RUN ln -sf /opt/conda/bin/python /usr/local/bin/python \
    && ln -sf /opt/conda/bin/python3 /usr/local/bin/python3


# ── System tools ──────────────────────────────────────────────────────────────
RUN rm -rf /var/lib/apt/lists/* && \
    apt-get clean && \
    apt-get update -o Acquire::Retries=3 -o Acquire::http::Pipeline-Depth=0 && \
    apt-get install -y --no-install-recommends -o Acquire::Retries=3 -o Acquire::http::Pipeline-Depth=0 \
        git \
        tmux \
        curl \
        unzip \
        openssh-server \
        vim \
        nano \
    && rm -rf /var/lib/apt/lists/*

# ── rclone (pinned) ───────────────────────────────────────────────────────────
ENV RCLONE_INSTALL_VERSION=v1.73.3
RUN curl -fsSL "https://github.com/rclone/rclone/releases/download/${RCLONE_INSTALL_VERSION}/rclone-${RCLONE_INSTALL_VERSION}-linux-amd64.zip" \
        -o /tmp/rclone.zip \
    && unzip /tmp/rclone.zip -d /tmp/rclone \
    && mv /tmp/rclone/rclone-*/rclone /usr/local/bin/rclone \
    && chmod +x /usr/local/bin/rclone \
    && rm -rf /tmp/rclone /tmp/rclone.zip

# ── Poetry ────────────────────────────────────────────────────────────────────
ENV POETRY_VERSION=2.2.1 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false

RUN curl -sSL https://install.python-poetry.org | python3 - --version ${POETRY_VERSION} \
    && ln -s /opt/poetry/bin/poetry /usr/local/bin/poetry \
    && poetry config virtualenvs.create false

# ── Workspace directories ─────────────────────────────────────────────────────
RUN mkdir -p \
    /workspace/repos \
    /workspace/runtime_configs \
    /workspace/command_logs

# ── SSH server setup ──────────────────────────────────────────────────────────
RUN mkdir -p /var/run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    && echo "PermitRootLogin yes" >> /etc/ssh/sshd_config \
    && echo "PasswordAuthentication no" >> /etc/ssh/sshd_config

EXPOSE 22


# ── Training image ────────────────────────────────────────────────────────────
FROM base AS training

# Copy lockfiles first — this layer is cached across code-only changes
WORKDIR /workspace/repos/scaling-llms

COPY pyproject.toml poetry.lock ./

RUN poetry install --no-root --only main --no-ansi

COPY . .

RUN poetry install --only main --no-ansi

COPY scripts/docker_entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]


# ── Dev image ─────────────────────────────────────────────────────────────────
FROM base AS dev

COPY scripts/docker_entrypoint_dev.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
CMD ["sleep", "infinity"]
