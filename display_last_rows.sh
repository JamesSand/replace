#!/bin/bash

logs_dir="logs"

if [ ! -d "$logs_dir" ]; then
    echo "Error: logs directory does not exist"
    exit 1
fi

# Check if there are any files in logs
if [ -z "$(ls -A $logs_dir 2>/dev/null)" ]; then
    echo "No log files found in $logs_dir"
    exit 0
fi

# Loop through all files in logs directory
for logfile in $logs_dir/*; do
    if [ -f "$logfile" ]; then
        echo "========================================"
        echo "File: $logfile"
        echo "========================================"
        tail -n 10 "$logfile"
        echo ""
    fi
done
