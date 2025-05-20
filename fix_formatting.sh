#!/bin/bash

# Create a log file to track which files have been processed
log_file="formatting_fixes.log"
echo "Starting formatting fixes at $(date)" > "$log_file"

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
    
    # Create a temporary prompt file for amp
    prompt_file=$(mktemp)
    cat > "$prompt_file" << EOF
Fix formatting error $error_code in $file_path at line $line_in_file, column $col_in_file.

Error details from the linter:
$error_context

Current code context:
$code_context

Return only the corrected code segment without explanations.
EOF
    
    # Execute amp call to fix the issue (pipe the prompt to amp)
    output_file=$(mktemp)
    amp < "$prompt_file" > "$output_file" 2>/dev/null
    
    # Check if amp produced any output
    if [ -s "$output_file" ]; then
        echo "Amp generated a fix. Inspect the output in $output_file"
        cat "$output_file" >> "$log_file"
    else
        echo "Amp did not produce any output for this error" >> "$log_file"
    fi
    
    # Mark completion in log
    echo "Completed fix attempt for $file_path:$line_in_file" >> "$log_file"
    echo "-----------------------------------"
    
    # Clean up temporary files
    rm -f "$prompt_file" "$output_file"
done

echo "All formatting issues have been processed. See $log_file for details."
