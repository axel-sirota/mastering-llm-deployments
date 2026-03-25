# Upload Week Materials to S3

Package and upload week notebooks to S3 for student distribution.

## Arguments

Takes one argument: the week number (e.g., `11`, `5-6-7`, `3-4`).
Optionally a second argument for the S3 folder display name (e.g., `LLMs`, `Sagemaker`).
If no display name is given, infer it from the exercise folder name.

## Instructions

### Step 1: Identify the files

Find all notebooks for the given week in both `exercises/` and `solutions/` directories:
```bash
ls exercises/week_XX_*/
ls solutions/week_XX_*/
```

### Step 2: Create the zip

Create a single zip file preserving directory structure (do NOT use `-j` flag — filenames collide between exercises and solutions):
```bash
zip zips/week_XX.zip \
    exercises/week_XX_topic/*.ipynb \
    solutions/week_XX_topic/*.ipynb
```

### Step 3: Upload to S3

Upload to the Bread Financial Academy folder on `courses.axel.net`:
```bash
aws s3 cp zips/week_XX.zip "s3://courses.axel.net/Bread Financial Academy/Week XX DisplayName/week_XX.zip"
```

### S3 Folder Naming Convention

Use the pattern `Week XX DisplayName/`:
- `Week 1-2 Pytorch/`
- `Week 3-4 Databricks Spark/`
- `Week 5-6-7 Sagemaker/`
- `Week 11 LLMs/`

### Step 4: Generate presigned URL

Generate a 7-day presigned URL and show it to the user:
```bash
aws s3 presign "s3://courses.axel.net/Bread Financial Academy/Week XX DisplayName/week_XX.zip" --expires-in 604800
```

### Step 5: Confirm

Show the user:
- Files included in the zip
- Zip size
- S3 location
- Public presigned URL (7-day expiry)

## Existing Uploads

| Week | S3 Folder | Zip |
|------|-----------|-----|
| 1-2 | Week 1-2 Pytorch/ | week_1-2_complete.zip |
| 3-4 | Week 3-4 Databricks Spark/ | week_3-4_complete.zip |
| 5-6-7 | Week 5-6-7 Sagemaker/ | week_5-6-7.zip |
| 11 | Week 11 LLMs/ | week_11.zip |
