"""Smoke tests — the sub-second feedback loop for S2.1.

The goal of these tests is a *fast* gate that catches "did this edit break the
import or the version?" in well under a second. They are intentionally tiny and
dependency-light; they do NOT exercise the acoustic model (that is S3.1+).
"""

import auditory_handshake


def test_package_imports() -> None:
    assert auditory_handshake is not None


def test_version_is_semver() -> None:
    version = auditory_handshake.__version__
    parts = version.split(".")
    assert len(parts) == 3, f"expected MAJOR.MINOR.PATCH, got {version!r}"
    assert all(part.isdigit() for part in parts), f"non-numeric in {version!r}"
