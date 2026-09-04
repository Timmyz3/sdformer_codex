#!/bin/bash
if [[ -z "${BASH_ENV:-}" && -z "${ENV:-}" ]]; then
    printf '%s\n' 'M641_BODY_SEES_EMPTY_STARTUP_HOOK_VARIABLES'
else
    printf '%s\n' 'M641_BODY_SEES_NONEMPTY_STARTUP_HOOK_VARIABLES'
fi
