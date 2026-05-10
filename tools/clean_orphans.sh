#!/bin/bash
# Copyright 2026- The Blackjax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Clean orphan Python processes that consumed 7.7 GB on 2026-05-10.
# See META-014 in /home/jp/blackjax-devs/WORKLOG.md for details.
#
# Kills:
# 1. Any `python.*sys;exec(eval` orphan (stdin-driven REPL from prior sessions)
# 2. Any pytest worker process (pgrep -f 'pytest|xdist') older than 5 minutes
#
# This script is idempotent and safe to re-run. Always exits 0 so make targets
# can chain it as a precondition without aborting.

set -e

KILLED=0
ORPHAN_COUNT=0

# Pattern 1: Kill stdin-driven REPL Python orphans.
# These hold memory and CPU indefinitely after the parent session dies.
if pgrep -f 'python.*sys;exec(eval' > /dev/null 2>&1; then
    REPL_PIDS=$(pgrep -f 'python.*sys;exec(eval' || true)
    if [ -n "$REPL_PIDS" ]; then
        echo "Killing stdin-driven REPL Python orphans:"
        while IFS= read -r pid; do
            if [ -n "$pid" ]; then
                echo "  PID $pid"
                pkill -9 -P "$pid" 2>/dev/null || true
                kill -9 "$pid" 2>/dev/null || true
                KILLED=$((KILLED + 1))
            fi
        done <<< "$REPL_PIDS"
    fi
fi

# Pattern 2: Kill stale pytest worker processes (older than 5 minutes).
# These can accumulate if test runs are interrupted, leaving workers hung.
FIVE_MIN_AGO=$(($(date +%s) - 300))
if pgrep -f 'pytest|xdist' > /dev/null 2>&1; then
    ps -o etime=,pid=,cmd= | grep -E 'pytest|xdist' | while read -r etime pid cmd; do
        if [ -n "$etime" ] && [ -n "$pid" ]; then
            # Parse etime; format is HH:MM:SS or MM:SS or S
            etime_secs=0
            if [[ "$etime" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
                # HH:MM:SS
                h="${BASH_REMATCH[1]}"
                m="${BASH_REMATCH[2]}"
                s="${BASH_REMATCH[3]}"
                # Use base 10 to avoid octal interpretation
                etime_secs=$((10#$h * 3600 + 10#$m * 60 + 10#$s))
            elif [[ "$etime" =~ ^([0-9]+):([0-9]+)$ ]]; then
                # MM:SS
                m="${BASH_REMATCH[1]}"
                s="${BASH_REMATCH[2]}"
                etime_secs=$((10#$m * 60 + 10#$s))
            elif [[ "$etime" =~ ^[0-9]+$ ]]; then
                # Just seconds (force base 10)
                etime_secs=$((10#$etime))
            fi

            # Kill if older than 5 minutes (300 seconds)
            if [ "$etime_secs" -gt 300 ]; then
                echo "Killing stale pytest worker PID $pid (age: $etime)"
                kill -9 "$pid" 2>/dev/null || true
                KILLED=$((KILLED + 1))
            fi
        fi
    done
fi

# Report outcome.
if [ "$KILLED" -eq 0 ]; then
    echo "no orphans found"
else
    echo "killed $KILLED orphan process(es)"
fi

exit 0
