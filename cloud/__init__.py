"""Cloud orchestration for the herbarium pipeline.

Modules:
    state         — per-project JSON state (pod id, volume id, dwca hash, ...)
    runpod_client — REST wrapper for the RunPod API

Higher-level layers (pod_session, orchestrator, secrets) will land here as
they're written; nothing in this package imports NiceGUI.
"""
