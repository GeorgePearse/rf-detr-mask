#!/bin/bash

# Create a log file to track which files have been processed
log_file="formatting_fixes.log"
echo "Starting formatting fixes at $(date)" > "$log_file"

# Create a directory to store the fixes
fixes_dir="formatting_fixes"
mkdir -p "$fixes_dir"

# Process each line in the formatting_failure.md file that starts with rfdetr/
grep -n '^rfdetr/' formatting_failure.md | while read -r line_with_number; do
    # Get line number in formatting_failure.md file
    md_line_number=$(echo "$line_with_number" | cut -d':' -f1)
    
    # Extract the full error line without the line number prefix
    error_line=$(echo "$line_with_number" | cut -d':' -f2- | sed 's/^[ \t]*//')
    
    # Parse the error line which is in format "rfdetr/path/to/file.py:123:45: E123 Error description"
    # First extract the full filepath up to the first number sequence (line number)
    file_path=$(echo "$error_line" | grep -o '^[^:]*')
    
    # Get line/column numbers and error code
    file_info=$(echo "$error_line" | sed 's/^[^:]*://')
    line_in_file=$(echo "$file_info" | cut -d':' -f1)
    col_in_file=$(echo "$file_info" | cut -d':' -f2 | sed 's/:.*//')
    error_code=$(echo "$error_line" | grep -o '[A-Z][0-9]\{3\}')
    
    # Skip if any required information is missing
    if [ -z "$file_path" ] || [ -z "$line_in_file" ] || [ -z "$error_code" ]; then
        echo "Skipping malformed line: $error_line" >> "$log_file"
        continue
    fi
    
    # Create a unique identifier for this error
    error_id="$(echo "$file_path" | tr '/' '_')_${line_in_file}_${col_in_file}_${error_code}"
    
    # Check if we've already processed this error
    if [ -f "$fixes_dir/$error_id.txt" ]; then
        echo "Already processed $file_path:$line_in_file:$col_in_file (Error: $error_code)" >> "$log_file"
        continue
    fi
    
    # Extract error description and context (several lines after the error line)
    context_start=$((md_line_number + 1))
    context_end=$((md_line_number + 15))
    error_context=$(sed -n "${context_start},${context_end}p" formatting_failure.md | grep -v "^$" | head -n 10)
    
    # Check if file exists and get the actual code context
    if [ -f "$file_path" ]; then
        start_line=$((line_in_file - 3))
        [ "$start_line" -lt 1 ] && start_line=1
        end_line=$((line_in_file + 5))
        code_context=$(sed -n "${start_line},${end_line}p" "$file_path")
    else
        echo "File not found: $file_path" | tee -a "$log_file"
        continue
    fi
    
    echo "==== Processing $file_path:$line_in_file:$col_in_file (Error: $error_code) ===="
    echo "Processing $file_path:$line_in_file:$col_in_file (Error: $error_code)" >> "$log_file"
    
    # Create a prompt file for amp with a 30-second timeout
    prompt_file="$fixes_dir/${error_id}_prompt.txt"
    cat > "$prompt_file" << EOF
Fix the following Python formatting error $error_code in $file_path at line $line_in_file, column $col_in_file:

Error details:
$error_context

Here's the current code:
$code_context

Return only the corrected code snippet, nothing else. No explanations.
EOF
    
    # Execute amp call with a timeout to prevent hanging
    output_file="$fixes_dir/${error_id}.txt"
    timeout 30s amp < "$prompt_file" > "$output_file" 2>/dev/null
    
    # Check if amp produced any output and the process didn't time out
    if [ $? -eq 0 ] && [ -s "$output_file" ]; then
        echo "✓ Amp generated a fix for $file_path:$line_in_file"
        echo "Fix saved to $output_file" >> "$log_file"
    else
        echo "✗ Failed to generate a fix for $file_path:$line_in_file"
        echo "Amp failed to fix $file_path:$line_in_file" >> "$log_file"
        # Create an empty file to mark this error as processed
        touch "$output_file"
    fi
    
    # Mark completion in log
    echo "Completed fix attempt for $file_path:$line_in_file" >> "$log_file"
    echo "-----------------------------------"
    
    # Small delay to prevent overwhelming the system
    sleep 1
done

echo "All formatting issues have been processed. See $log_file for details."
echo "Fixes are stored in the $fixes_dir directory."
