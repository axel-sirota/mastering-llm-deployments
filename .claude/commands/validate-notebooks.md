# Validate Notebooks

Run comprehensive validation on exercise and/or solution notebooks.

## Instructions

You should run the comprehensive notebook validation script to verify:
- Python syntax correctness
- Import availability
- Exercise notebooks have proper `= None` placeholders
- Solution notebooks have complete implementations
- Paired notebooks match structurally
- Generate requirements.txt if needed

## Usage Patterns

### After building exercise notebook
```bash
python validate_notebooks.py exercises/week_XX_topic/week_XX_topic.ipynb --type exercise
```

### After building solution notebook
```bash
python validate_notebooks.py solutions/week_XX_topic/week_XX_topic.ipynb --type solution
```

### After completing both notebooks
```bash
python validate_notebooks.py --pair \
    exercises/week_XX_topic/week_XX_topic.ipynb \
    solutions/week_XX_topic/week_XX_topic.ipynb
```

### Generate requirements.txt
```bash
python validate_notebooks.py exercises/week_XX_topic/week_XX_topic.ipynb --requirements
```

## What to Check

✅ **All validations pass** - No errors reported
✅ **Exercise has placeholders** - All lab cells show `= None` patterns
✅ **Solution is complete** - No `= None` in lab cells
✅ **Structure matches** - Same cell count and types
✅ **Requirements generated** - requirements.txt created if needed

## When to Run

- After creating/editing exercise notebook
- After creating/editing solution notebook
- Before committing notebooks to git
- Before distributing notebooks to students

## See Also

- `initial_docs/VALIDATION.md` - Complete validation documentation
- `validate_notebooks.py` - The validation script
