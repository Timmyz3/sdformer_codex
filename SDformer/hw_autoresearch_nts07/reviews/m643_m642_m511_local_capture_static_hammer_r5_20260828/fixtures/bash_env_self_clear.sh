printf '%s\n' 'M643_BASH_ENV_EXECUTED_BEFORE_BODY'
unset BASH_ENV ENV
m643_hook_function() { printf '%s\n' 'M643_HOOK_FUNCTION_CALLED'; }
