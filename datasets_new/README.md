# Datasets: Structure and Scripts (`datasets_new`)

This folder contains all resources, scripts, and schemas for building, structuring, and exporting training datasets for fine-tuning language models. Below is a detailed guide to the directory's structure, its main files, and how to use them.

---

## Directory Structure

```
datasets_new/
├── build_train_jsonl.py
├── create_dataset_by_structure.py
├── structure.enriched.json
├── train_data.jsonl
├── schemas/
│   └── schema_reasoning.json
└── topics/
    ├── ai/
    ├── business/
    ├── ethics/
    ├── finance/
    ├── geography/
    ├── marketing/
    ...
```

---

## Main Files and Their Roles

### 1. `structure.enriched.json`

-   **Purpose:**

    -   Defines the full taxonomy of dataset topics, subcategories, and their metadata.
    -   Each top-level key is a category (e.g., `ai`, `business`, `ethics`), containing subcategories with descriptions, example questions, content types, and tags.

-   **Usage:**

    -   Used by scripts to determine which topics and subcategories to process, and to enrich each data sample with consistent metadata.

-   **Example Entry:**

    ```json
    "ai": {
      "coding_development": {
        "description": "AI for software engineering: code generation, refactoring, testing, and security.",
        "example_questions": [ ... ],
        "content_type": ["reasoning"],
        "tags": ["programming", "devtools", "testing", "security"]
      }
    }
    ```

### 2. `schemas/`

-   **Purpose:**

    -   Contains JSON schema files that define the expected structure for dataset entries.
    -   Currently, all datasets follow a **reasoning schema** with both `think` (reasoning) and `output` (final answer).

-   **Usage:**

    -   Use the schema to validate dataset files or as a reference for formatting new data.

### 3. `topics/`

-   **Purpose:**

    -   Contains all raw data files, organized by category and subcategory.
    -   Each subcategory folder contains files named as `{category}.{subcategory}.reasoning.json` (e.g., `ai.coding_development.reasoning.json`).

-   **Usage:**

    -   These files are the source data for dataset building scripts.

### 4. `build_train_jsonl.py`

-   **Purpose:**

    -   Main script to flatten and export all topic data into a single JSONL file (`train_data.jsonl`), with per-sample metadata.

-   **How it works:**

    -   Loads the structure from `structure.enriched.json`.
    -   Iterates over all categories, subcategories, and content types.
    -   Reads each corresponding topic file, normalizes entries, and enriches them with metadata (category, subcategory, tags, etc.).
    -   Ensures IDs start from `1` **per file** (not globally).
    -   Outputs a single `train_data.jsonl` file, ready for training or further processing.

-   **Usage:**

    ```sh
    python build_train_jsonl.py --structure structure.enriched.json --topics-dir topics --output train_data.jsonl
    ```

### 5. `create_dataset_by_structure.py`

-   **Purpose:**

    -   (If present) Used for generating or organizing topic files according to the structure defined in `structure.enriched.json`.
    -   May help automate the creation of empty or template files for new topics/subcategories.

-   **Usage:**

    -   Refer to the script's help or source for details.

### 6. `train_data.jsonl`

-   **Purpose:**

    -   The final, flattened dataset file produced by `build_train_jsonl.py`.
    -   Each line is a JSON object with all required fields and metadata.

---

## Workflow: Building a Training Dataset

1. **Edit or add topic data:**

    - Place new or updated data files in the appropriate `topics/{category}/{subcategory}/` folder, following the naming convention.

2. **Update structure:**

    - If adding new topics or subcategories, update `structure.enriched.json` accordingly.

3. **Validate data (optional):**

    - Use the schema in `schemas/` to check your data files for correctness.

4. **Build the dataset:**

    - Run `build_train_jsonl.py` to generate `train_data.jsonl`.

5. **Use `train_data.jsonl` for training or further processing.**

---

## Dataset Philosophy

-   All entries include both:

    -   **`think`**: the reasoning or step-by-step thought process.
    -   **`output`**: the final answer, concise and user-facing.

-   Difficulty levels (1–5) now handle the gradient from **very simple reasoning** to **deep, multi-step expert reasoning**.
-   This makes a separate QA category unnecessary; instead, **difficulty 1–2 covers short/light reasoning**, and **3–5 covers complex reasoning**.

---

## Tips & Best Practices

-   Always keep `structure.enriched.json` in sync with the actual topics and subcategories present in `topics/`.
-   Ensure every entry contains both `think` and `output`.
-   Tags and metadata are automatically merged and deduplicated by the export script.

---

## Troubleshooting: JSON type consistency

If you see errors like:

```
pyarrow.lib.ArrowInvalid: JSON parse error: Column(/difficulty) changed from string to number in row 150
```

This means one or more rows in `train_data.jsonl` have inconsistent types for the same field (commonly `difficulty`). The data loader (pandas/pyarrow) requires consistent column types across rows.

`build_train_jsonl.py` coerces `difficulty` into an integer (mapping common text values like `easy/medium/hard`) to keep types stable.

Quick fix script example is included in this repo (`README` previously).

---

## See Also

-   For more details on the data format, see the schema file in `schemas/`.
-   For advanced usage or troubleshooting, read the source code of the scripts or use the `--help` flag.
