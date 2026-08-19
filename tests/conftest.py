"""Shared pytest configuration.

Phase 6.1 makes ``Settings`` a pure Pydantic model. Tests no longer need to
mutate an ``env_file`` setting or clear a BaseSettings cache: process
environment and developer ``.env`` files are outside the Settings contract.
"""
