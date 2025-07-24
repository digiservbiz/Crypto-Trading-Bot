<#
.SYNOPSIS
    Fixes missing Git folders while preserving ignore rules
#>

# 1. Create directories if they don't exist
if (!(Test-Path "models")) { New-Item -ItemType Directory -Path "models" }
if (!(Test-Path "data")) { New-Item -ItemType Directory -Path "data" }

# 2. Create .gitkeep files
@"
# Git placeholder to maintain folder structure
# Delete this file when adding real models/data
"@ | Out-File "models\.gitkeep" -Encoding utf8

@"
# Git placeholder to maintain folder structure
# Delete this file when adding real data
"@ | Out-File "data\.gitkeep" -Encoding utf8

# 3. Create .gitignore files
@"
# Ignore everything except whitelisted files
*
!.gitkeep
!*.json
!*.yaml
!*.pkl
"@ | Out-File "models\.gitignore" -Encoding utf8

@"
# Ignore all data except these patterns
*
!.gitkeep
!symbols.csv
!historical/*.parquet
"@ | Out-File "data\.gitignore" -Encoding utf8

# 4. Update root .gitignore
@"
# Root .gitignore additions
/models/*
!/models/.gitkeep
!/models/.gitignore

/data/*
!/data/.gitkeep
!/data/.gitignore

# Sensitive files (always exclude)
.env
*.key
*.secret
"@ | Add-Content ".gitignore"

# 5. Stage and commit
git add models/.gitkeep models/.gitignore
git add data/.gitkeep data/.gitignore
git add .gitignore

git commit -m "Fix missing folders with proper git tracking"
git push