#!/bin/bash -p
if [[ -n "${BASH_ENV:-}" || -n "${ENV:-}" ]]; then
    printf '%s\n' 'M643_BODY_SEES_NONEMPTY_STARTUP_HOOK_VARIABLES'
else
    printf '%s\n' 'M643_BODY_SEES_EMPTY_STARTUP_HOOK_VARIABLES'
fi
if type m643_exported_attack >/dev/null 2>&1; then
    printf '%s\n' 'M643_EXPORTED_FUNCTION_IMPORTED'
else
    printf '%s\n' 'M643_EXPORTED_FUNCTION_NOT_IMPORTED'
fi
printf '%s\n' 'M643_PRIVILEGED_VICTIM_BODY_EXECUTED'
