#!/bin/sh
# Picks the model for the server:
# - DLLM_MODEL set: that file is served as it is.
# - Otherwise $DLLM_MODEL_FILE (default /models/model.dllm) is served when it exists. When it does not and
#   DLLM_IMPORT names a source (for example hf:HuggingFaceTB/SmolLM2-135M-Instruct), that model is imported into it
#   first (DLLM_IMPORT_ARGS adds options such as --licence or --repo); mount /models as a volume to
#   keep it.
# - Neither: the built-in placeholder model.
set -e
if [ -z "$DLLM_MODEL" ]; then
    if [ ! -f "$DLLM_MODEL_FILE" ] && [ -n "$DLLM_IMPORT" ]; then
        echo "dllm-entrypoint: importing $DLLM_IMPORT into $DLLM_MODEL_FILE" >&2
        # shellcheck disable=SC2086  # DLLM_IMPORT_ARGS is split into options on purpose
        dllm import "$DLLM_IMPORT" -o "$DLLM_MODEL_FILE" --cache /models/.cache $DLLM_IMPORT_ARGS
    fi
    if [ -f "$DLLM_MODEL_FILE" ]; then
        export DLLM_MODEL="$DLLM_MODEL_FILE"
    fi
fi
exec "$@"
