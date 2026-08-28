#!/usr/bin/env python3
"""Start an SSH-only NInfer development container for interactive diagnosis."""

from __future__ import annotations

import os
import pathlib


def configure_ssh_key() -> None:
    public_key = os.getenv("PUBLIC_KEY", "").strip()
    if not public_key:
        raise SystemExit("PUBLIC_KEY is required for the debug image")

    ssh_dir = pathlib.Path("/root/.ssh")
    ssh_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    ssh_dir.chmod(0o700)
    authorized_keys = ssh_dir / "authorized_keys"
    authorized_keys.write_text(public_key + "\n", encoding="utf-8")
    authorized_keys.chmod(0o600)


def main() -> None:
    configure_ssh_key()
    pathlib.Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    os.system("ssh-keygen -A")
    os.execv(
        "/usr/sbin/sshd",
        ["/usr/sbin/sshd", "-D", "-e", "-o", "PasswordAuthentication=no"],
    )


if __name__ == "__main__":
    main()
