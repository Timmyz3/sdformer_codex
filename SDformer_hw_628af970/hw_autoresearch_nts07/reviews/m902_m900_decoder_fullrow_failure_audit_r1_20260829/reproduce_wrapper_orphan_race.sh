#!/bin/bash -p
set -u
PATH=/usr/bin:/bin
export PATH

# Bounded process-control reproduction only.  No M896/M900 workload is read.
# It intentionally leaves two tiny moved directories under /tmp so the test
# never deletes a broad or unresolved path.

m902_broken_dir="$(mktemp -d /tmp/m902_broken_wrapper.XXXXXX)"
export M902_BROKEN_DIR="${m902_broken_dir}"
m902_broken_worker() {
    /usr/bin/env -i PATH=/usr/bin:/bin M902_BROKEN_DIR="${M902_BROKEN_DIR}" \
        /usr/bin/python3 -c \
        'import os,time,pathlib; time.sleep(1); pathlib.Path(os.environ["M902_BROKEN_DIR"], "late_write").touch()'
}
m902_broken_worker &
m902_wrapper_pid=$!
sleep 0.2
m902_python_pid="$(pgrep -P "${m902_wrapper_pid}" | head -1)"
kill -TERM "${m902_wrapper_pid}"
wait "${m902_wrapper_pid}"
m902_wrapper_rc=$?
m902_orphan_after_wait=no
kill -0 "${m902_python_pid}" 2>/dev/null && m902_orphan_after_wait=yes
mv "${M902_BROKEN_DIR}" "${M902_BROKEN_DIR}.moved"
sleep 1.2
m902_broken_late_write=no
[[ -e "${M902_BROKEN_DIR}.moved/late_write" ]] && m902_broken_late_write=yes

m902_fixed_dir="$(mktemp -d /tmp/m902_direct_worker.XXXXXX)"
export M902_FIXED_DIR="${m902_fixed_dir}"
/usr/bin/env -i PATH=/usr/bin:/bin M902_FIXED_DIR="${M902_FIXED_DIR}" \
    /usr/bin/python3 -c \
    'import os,time,pathlib; time.sleep(1); pathlib.Path(os.environ["M902_FIXED_DIR"], "late_write").touch()' &
m902_direct_pid=$!
sleep 0.2
kill -TERM "${m902_direct_pid}"
wait "${m902_direct_pid}"
m902_direct_rc=$?
m902_direct_alive_after_wait=no
kill -0 "${m902_direct_pid}" 2>/dev/null && m902_direct_alive_after_wait=yes
mv "${M902_FIXED_DIR}" "${M902_FIXED_DIR}.moved"
sleep 1.2
m902_fixed_late_write=no
[[ -e "${M902_FIXED_DIR}.moved/late_write" ]] && m902_fixed_late_write=yes

printf 'broken_wrapper_rc=%s orphan_after_wait=%s late_write_after_rename=%s\n' \
    "${m902_wrapper_rc}" "${m902_orphan_after_wait}" "${m902_broken_late_write}"
printf 'direct_worker_rc=%s alive_after_wait=%s late_write_after_rename=%s\n' \
    "${m902_direct_rc}" "${m902_direct_alive_after_wait}" "${m902_fixed_late_write}"

[[ "${m902_wrapper_rc}" -eq 143 && \
   "${m902_orphan_after_wait}" == yes && \
   "${m902_broken_late_write}" == no && \
   "${m902_direct_rc}" -eq 143 && \
   "${m902_direct_alive_after_wait}" == no && \
   "${m902_fixed_late_write}" == no ]]
printf 'PASS_M902_BOUNDED_WRAPPER_ORPHAN_RACE_REPRODUCTION\n'
