# Set strict mode and exit on error
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Define variables
$NUM_JOB ="36"
$BASE_DIR = "C:/Users/bezem/Documents/erdos_deep_learning/Speech-Editing-Toolkit-stable-unedited/Speech-Editing-Toolkit-stable/data/processed/libritts"
$MODEL_NAME = "mfa_model"
$MFA_INPUTS =  "mfa_inputs"
$MFA_OUTPUTS = "mfa_outputs"


# Remove existing output directory
#Remove-Item -Recurse -Force "$BASE_DIR/mfa_outputs_tmp"

# Execute MFA alignment

mfa align "$BASE_DIR/$MFA_INPUTS" "$BASE_DIR/mfa_dict.txt" "$BASE_DIR/$MODEL_NAME.zip" "$BASE_DIR/mfa_tmp"  --clean -j $NUM_JOB `
    --config_path "C:/Users/bezem/Documents/erdos_deep_learning/Speech-Editing-Toolkit-stable-unedited/Speech-Editing-Toolkit-stable/data_gen/tts/mfa_train_config.yaml"


# Create output directory and move files
New-Item -ItemType Directory -Force -Path "$BASE_DIR/$MFA_OUTPUTS"

Get-ChildItem -Path "$BASE_DIR/mfa_outputs_tmp" -Filter *.TextGrid -Recurse | ForEach-Object {
    Move-Item $_.FullName "$BASE_DIR/$MFA_OUTPUTS/"
}

# Check if unaligned.txt exists and copy it
if (Test-Path "$BASE_DIR/mfa_outputs_tmp/unaligned.txt") {
    Copy-Item "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" "$BASE_DIR/"
}

# Cleanup temporary directories
Remove-Item -Recurse -Force "$BASE_DIR/mfa_outputs_tmp"
Remove-Item -Recurse -Force "$BASE_DIR/mfa_tmp"

# Create the final output directory and sync files
New-Item -ItemType Directory -Force -Path "$BASE_DIR/mfa_outputs"
Get-ChildItem -Path "$BASE_DIR/mfa_tmp/mfa_inputs_train_acoustic_model/sat_2_ali/textgrids" `
    -Recurse -Depth 1 | Where-Object { $_.PSIsContainer } | ForEach-Object {
        robocopy $_.FullName "$BASE_DIR/mfa_outputs/" /E
    }

# Copy unaligned.txt again if it exists
if (Test-Path "$BASE_DIR/mfa_outputs_tmp/unaligned.txt") {
    Copy-Item "$BASE_DIR/mfa_outputs_tmp/unaligned.txt" "$BASE_DIR/"
}

# Final cleanup
Remove-Item -Recurse -Force "$BASE_DIR/mfa_outputs_tmp"
Remove-Item -Recurse -Force "$BASE_DIR/mfa_tmp"