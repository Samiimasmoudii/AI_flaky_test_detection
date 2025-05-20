## Load Configuration and Constants:

imports RESULTS_DIR and SAMPLE_SIZE from inc.config to define the output directory for results and the number of samples to process.

install requirements using 
``` pip install -r requirements.txt ```

## Load and Clean Data:

Use the load_and_clean_csv function from inc.data_loader to load and clean the CSV file (py-data.csv) containing repository data.

## Fetch Changed Files from GitHub:

It uses the get_changed_files and download_file_content functions from inc.github_utils to:
Retrieve the list of files changed in a specific commit.
Download the content of those files for both the current and parent commits.


## Save File Versions and Generate Diffs:

It uses the save_versions function from inc.processor to:
Save the "before" and "after" versions of the changed files.
Generate a unified diff (developer_patch.diff) between the two versions.


## Run LLM Analysis:

It uses the run_llm_on_file function from inc.processor to analyze the "before" version of the file using the Gemini LLM and generate a suggested fix for flaky test issues.
Compare Fixes:

It uses the compare_fixes function from inc.processor to compare the LLM-generated fix with the developer's patch and save the comparison in a comparison.txt file.
Process Each Repository:

The process_repo function iterates over the repositories listed in the CSV file, processes each repository's data, and saves the results in the RESULTS_DIR.
Error Handling and Logging:

If any errors occur during processing, they are logged to an error_log.txt file.

## Main Execution:

The main function ensures the required directories exist, loads the CSV file, and processes a random sample of repositories based on the SAMPLE_SIZE.