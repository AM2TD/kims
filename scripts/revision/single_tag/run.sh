#!/bin/bash
# Single-tag NTU test. 28 KIMS runs, 14 at a time (GPU limit), stopped as a whole
# if any run diverges (watch.py). Then: python3 score.py plan.txt table.txt ; python3 fig.py plan.txt z out.png
H=$(cd "$(dirname "$0")" && pwd); cd "$(dirname "$0")/../../.."
python3 $H/make_jobs.py
(xargs -P 14 -I{} bash -c "{}" < $H/jobs.txt > /dev/null 2>&1 & echo $! > $H/runner.pid; wait) &
sleep 2; python3 $H/watch.py $(cat $H/runner.pid)
