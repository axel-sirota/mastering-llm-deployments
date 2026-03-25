---
description: Build exercise and solution notebooks for a specific week
---

Build notebooks for week: $ARGUMENTS

**CRITICAL INSTRUCTIONS - READ CAREFULLY BEFORE STARTING**

## Command Arguments

This command takes TWO arguments:
1. **Week number** (e.g., `2`, `11`, `17`)
2. **Environment** (one of: `colab`, `databricks`, `sagemaker`, `localhost`)

Example usage: `/build-week-notebook 2 colab`

---

## Pre-Work: MANDATORY Reading & Planning

### Step 1: Read All Context Files (DO THIS FIRST)

**YOU MUST READ THESE FILES BEFORE DOING ANYTHING:**

1. **[CLAUDE.md](CLAUDE.md:1-385)** - Teaching philosophy, notebook structure, tone guidelines
2. **[initial_docs/outline.md](initial_docs/outline.md:1-809)** - Week content, learning objectives, labs
3. **[initial_docs/technical_speecs.md](initial_docs/technical_speecs.md:1-652)** - Environment setup specs
4. **[exercises/week_01_pytorch_basics/NOTEBOOK_GUIDE.md](exercises/week_01_pytorch_basics/NOTEBOOK_GUIDE.md:1-569)** - Example of proper cell-by-cell structure
5. **[exercises/week_01_pytorch_basics/week_01_pytorch_basics.ipynb](exercises/week_01_pytorch_basics/week_01_pytorch_basics.ipynb:1-1)** - Reference for structure
6. **[solutions/week_01_pytorch_basics/week_01_pytorch_basics.ipynb](solutions/week_01_pytorch_basics/week_01_pytorch_basics.ipynb:1-1)** - Reference for solutions

### Step 2: Extract Week Information from Outline

From [initial_docs/outline.md](initial_docs/outline.md:1-809), extract for the requested week:
- Week title and theme
- "Students arrive with" (prerequisites)
- "We teach/practice" (main topics)
- "Labs" (hands-on exercises - THESE ARE THE CORE)
- "Skills developed" (learning outcomes)
- "Extra/Optional (async)" (advanced challenges)

### Step 3: Plan the Notebook Structure

**IN CHAT, SHOW ME YOUR PLAN:**

List out:
1. Week title and learning objectives (derived from outline)
2. Real-world context/storytelling angle (CRITICAL - every week needs a compelling story)
3. List of 2-4 main topics (break down "We teach/practice")
4. For EACH topic, show:
   - Theory markdown (what concepts to cover)
   - Demo code (simple focused example)
   - Lab instructions (what students will build)
   - Starter code structure
5. Optional/Extra labs (from "Extra/Optional" in outline)
6. Environment setup (based on environment argument)

**DO NOT PROCEED until I approve this plan.**

---

## Core Teaching Principles (from CLAUDE.md)

### 1. Storytelling & Real-World Context
- **EVERY topic** starts with a real problem or business scenario
- Connect concepts to practical applications
- Frame public datasets in realistic contexts
- Example: "You're a data scientist at a bank processing thousands of checks daily..."

### 2. Heavy Documentation
- **Code cells**: Every line has meaningful comments (what AND why)
- **Markdown cells**: Detailed explanations with embedded code examples using ` ```python ``` ` blocks
- **Theory sections**: Clear but concise (students watched videos)
- **Lab instructions**: Step-by-step, detailed enough for independent work

### 3. Demo-Driven Learning (Show, then Do)
- Demos are simple, focused examples
- Showcase ONE concept clearly
- Students see it done, then do it themselves in labs
- Pattern: **Theory → Demo → Lab**
- It is extremely important for every markdown cell we introduce some theory ("Example, what is Convolution layer?") automatically next cell is code cell, even one line showing it in action. Dont chain more than 3 markdown cells of concepts without a code cell in the middle, at least showing, even if not real full demo.

### 4. Appropriate Difficulty
- Labs are **medium difficulty**: achievable in 15-30 minutes
- Assumes students watched pre-class videos
- Focus on application and muscle memory
- Optional/extra labs provide challenge

### 5. Public Datasets Only
- MNIST, CIFAR-10, Iris, scikit-learn datasets, HuggingFace datasets
- Fetch from public URLs (torchvision.datasets, sklearn.datasets, etc.)
- Document dataset sources clearly

### 6. Tone: Friendly but Professional
- Conversational without being overly casual
- Encouraging and supportive
- Clear, direct instructions
- No jargon unless explained
- Don't mention specific Bread financial situations, because we don't know how it will apply to them.

---

## Notebook Structure Template

Each week MUST follow this structure:

### Header Section
1. **Cell 0 (Markdown)**: Week title, learning objectives, prerequisites, session format, GPU setup (if Colab)
2. **Cell 1 (Markdown)**: "Section 0: Environment Setup" header
3. **Cell 2 (Code)**: Package installation (environment-specific)
4. **Cell 3 (Code)**: Import libraries, verify environment, set random seeds
5. **Cell 4 (Markdown)**: "What Are We Building Today?" - Real-world context/storytelling
6. **Cell 5 (Code)**: Preview of dataset or final outcome

### For Each Topic (2-4 topics per week)
1. **Markdown Cell**: Topic title + theory introduction
   - Real-world context paragraph
   - Concept explanation with inline code examples (use ` ```python ``` `)
   - "Demo: [Topic Name]" section
2. **Code Cell(s)**: Demo code (complete, heavily commented)
3. **Markdown Cell**: Lab instructions (detailed, step-by-step)
4. **Code Cell**: Lab starter code with `None # YOUR CODE` placeholders

### Closing Section
1. **Markdown Cell**: Optional/Extra labs (clearly marked)
2. **Code Cell(s)**: Optional lab starter code
3. **Markdown Cell**: Congratulations, what you learned, key takeaways, next steps, resources

---

## CRITICAL: Incremental Building - MANDATORY APPROVAL CHECKPOINTS

**⚠️ YOU MUST NEVER ADD MORE THAN 5 CELLS WITHOUT EXPLICIT USER APPROVAL**

### MANDATORY Process (DO NOT DEVIATE):

1. **Add maximum 5 cells**
2. **STOP IMMEDIATELY**
3. **Ask user**: "I've added cells X-Y. How does it look? Should I continue?"
4. **DO NOT PROCEED** until user explicitly approves with words like "continue", "yes", "good", or "approved"
5. **If user says "continue"**: Return to step 1 for next batch of 5 cells ONLY
6. **NEVER interpret "continue" as permission to do ALL remaining cells**

### Violation Prevention:

**❌ FORBIDDEN**: Adding 6+ cells without asking for approval
**❌ FORBIDDEN**: Interpreting "continue" as "do everything"
**❌ FORBIDDEN**: "Getting carried away with momentum"
**❌ FORBIDDEN**: Assuming you can skip approval checkpoints

**✅ REQUIRED**: Stop every 5 cells, no exceptions
**✅ REQUIRED**: Wait for explicit approval each time
**✅ REQUIRED**: Treat this as a hard checkpoint, not a suggestion
**✅ REQUIRED**: After user says "continue", add ONLY the next 5 cells, then STOP again

### Why This Matters:
- Jupyter notebooks are JSON files that become HUGE very quickly
- User maintains control and can review quality at each stage
- Prevents file size issues and missing content
- Catches errors early before they propagate
- Writing entire notebooks at once leads to:
  - Files too large to manage
  - Missing content due to truncation
  - Difficult to review and debug
  - Version control nightmares

### Implementation Example:

Use NotebookEdit tool to add cells incrementally:

```
First batch (cells 0-4):
- Cell 0: Header markdown
- Cell 1: Environment setup markdown
- Cell 2: Package installation code
- Cell 3: Imports and verification code
- Cell 4: Real-world context markdown

STOP - Ask user: "I've added cells 0-4. How does it look? Should I continue?"
WAIT FOR APPROVAL

**VALIDATION CHECKPOINT**: Run basic validation after every 5 cells:
```bash
python validate_notebooks.py exercises/week_XX_topic/week_XX_topic.ipynb --type exercise
```
This ensures:
- Python syntax is valid
- No indentation errors
- Cells are structured correctly

Second batch (cells 5-9):
- Cell 5: Dataset preview code
- Cell 6: Topic 1 theory markdown
- Cell 7: Topic 1 demo code
- Cell 8: Topic 1 lab instructions markdown
- Cell 9: Topic 1 lab starter code

STOP - Ask user: "I've added cells 5-9. How does it look? Should I continue?"
WAIT FOR APPROVAL

**VALIDATION CHECKPOINT**: Run validation again

... repeat this cycle until notebook is complete
```

---

## Markdown Rendering (CRITICAL)

**COMMON MISTAKE**: Markdown not rendering correctly

### ✅ CORRECT Markdown Cell Format:

```python
# When using NotebookEdit with cell_type="markdown":
new_source = """# Week 1: PyTorch Basics

## Learning Objectives

By the end of this session, you will:
- Work with PyTorch tensors
- Understand autograd
- Build neural networks

### Example Code Block

```python
import torch
x = torch.tensor([1, 2, 3])
```

**Bold text** and *italic text* work fine.
"""
```

### ❌ WRONG - Do NOT do this:

```python
# Don't escape markdown
new_source = "\\# Week 1\\n\\n\\*\\*Bold\\*\\*"  # WRONG

# Don't use triple quotes inside triple quotes incorrectly
new_source = """```python code```"""  # May break - use backticks carefully
```

### Testing Markdown:
After adding markdown cells, verify:
- Headers render as headers (# ## ###)
- Bold/italic render correctly (**bold** *italic*)
- Code blocks render with syntax highlighting
- Lists render properly
- Links work

---

## Environment-Specific Setup

### Colab Environment

**Package Installation Cell:**
```python
# Install required packages (run this first in Google Colab)
# If running locally with conda/venv, you may skip this cell

!pip install [packages based on week]
```

**Import Cell:**
```python
# Import all necessary libraries
import [libraries]

# Check versions
print(f"Library version: {library.__version__}")

# Device configuration - automatically use GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

if device.type == 'cuda':
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

# Set random seed for reproducibility
torch.manual_seed(42)

print("\n✅ Environment setup complete!")
```

### Databricks Environment

**Package Installation Cell:**
```python
# Install additional packages not in Databricks ML Runtime
%pip install [additional packages]
```

**Import Cell:**
```python
# Import libraries
import [libraries]

# Databricks comes with Spark pre-configured
print(f"Spark version: {spark.version}")

# Verify libraries
print(f"Library version: {library.__version__}")

print("\n✅ Environment setup complete!")
```

### SageMaker Environment

**Package Installation Cell:**
```python
# Install required packages
!pip install [packages]
```

**Import Cell:**
```python
# Import libraries
import [libraries]
import sagemaker
from sagemaker import get_execution_role

# SageMaker session
sess = sagemaker.Session()
role = get_execution_role()
bucket = sess.default_bucket()

print(f"SageMaker version: {sagemaker.__version__}")
print(f"Role: {role}")
print(f"Bucket: {bucket}")

print("\n✅ Environment setup complete!")
```

### Localhost Environment

**Package Installation Cell:**
```python
# Ensure you're in your virtual environment
# Run: python -m venv academy-env
# Activate: source academy-env/bin/activate (macOS/Linux)
#        or academy-env\\Scripts\\activate (Windows)

# Install packages (run in terminal):
# pip install [packages]

# This cell is informational - run the pip commands in your terminal
```

**Import Cell:**
```python
# Import libraries
import [libraries]

# Verify installations
print(f"Library version: {library.__version__}")

print("\n✅ Environment setup complete!")
```

---

## Exercise vs Solution Notebooks

### Exercise Notebook (exercises/week_XX_topic/)

**Lab starter cells should have:**
```python
# Your code here

# 1. [Task description]
variable_1 = None  # YOUR CODE

# 2. [Task description]
variable_2 = None  # YOUR CODE

# Verification (provide this)
if variable_1 is not None:
    print(f"Result: {variable_1}")
```

**Characteristics:**
- Full theory and explanations
- Complete demo code
- Detailed lab instructions
- Starter code with `None # YOUR CODE` placeholders
- Verification/test code (so students know if they're right)

### Solution Notebook (solutions/week_XX_topic/)

**Lab solution cells should have:**
```python
# Solution: Lab X.Y - [Lab Title]

# 1. [Task description]
variable_1 = actual_implementation  # Complete implementation with comment

# 2. [Task description]
variable_2 = actual_implementation  # Complete implementation with comment

# Verification
if variable_1 is not None:
    print(f"Result: {variable_1}")

# Explanation:
# [Detailed explanation of the solution approach]
# [Common mistakes students make]
# [Alternative approaches]
```

**Characteristics:**
- Everything from exercise notebook
- Fully completed lab code
- Extensive explanatory comments
- Expected outputs visible (can include output in cells)
- Notes on common mistakes or alternatives

---

## Pedagogical Flow (CRITICAL)

**All labs must build toward the week's main outcome.**

From outline, identify the PRIMARY lab (the main skill). Example:

Week 1 outline says:
- **Labs**: "MNIST digit classification with feedforward network"
- **Skills developed**: "Can build and train basic neural networks in PyTorch"

This means:
- **Topic 1 (Tensors)**: Build foundation for representing images as tensors
  - Demo: Create tensors, operations, reshaping
  - Lab: Practice tensor operations needed for neural networks
- **Topic 2 (Autograd)**: Understand how gradients work for training
  - Demo: Simple gradient computation
  - Lab: Compute gradients for functions
- **Topic 3 (nn.Module)**: Build the actual network
  - Demo: Simple network architecture
  - Lab: Build deeper network
- **Topic 4 (Training Loop)**: Put it all together
  - Demo: Complete training loop
  - **Lab: MAIN OUTCOME - Train MNIST classifier** ← This is the goal!
- **Topic 5 (Evaluation)**: Assess what we built
  - Demo: Compute accuracy, confusion matrix
  - Lab: Analyze the model we trained

Each topic builds toward the final outcome. Students can't do Topic 4 without Topics 1-3.

### Extra/Optional Labs

These are CRITICAL for:
- Fast finishers (keep them engaged)
- Advanced students (challenge them)
- Async learning (students practice at home)

Must include:
- Clearly marked as "Optional" or "Extra"
- Different difficulty levels
- Relate to the week's theme but explore deeper
- Example from Week 1: "Iris dataset classification" (same concepts, different dataset)

---

## Workflow Summary

1. **Read all context files** (mandatory)
2. **Extract week info from outline** and show me
3. **Plan notebook structure** and get my approval
4. **Create directories**: `exercises/week_XX_topic/` and `solutions/week_XX_topic/`
5. **Build EXERCISE notebook incrementally:**
   - Create skeleton
   - Add 5 cells at a time
   - **VALIDATION CHECKPOINT**: Run `python validate_notebooks.py exercises/week_XX_topic/week_XX_topic.ipynb --type exercise` after every 5 cells
   - After each batch, STOP and ask: "I've added cells X-Y. How does it look? Should I continue?"
   - Wait for approval
   - Continue until complete
6. **Build SOLUTION notebook incrementally:**
   - Copy structure from exercise
   - Fill in solutions
   - Add explanatory comments
   - **VALIDATION CHECKPOINT**: Run `python validate_notebooks.py solutions/week_XX_topic/week_XX_topic.ipynb --type solution` after every 5 cells
   - Add 5 cells at a time with approval
7. **Final verification:**
   - Run comprehensive validation: `python validate_notebooks.py --pair exercises/.../notebook.ipynb solutions/.../notebook.ipynb`
   - Generate requirements.txt: `python validate_notebooks.py exercises/.../notebook.ipynb --requirements`
   - Both notebooks run top-to-bottom without errors
   - Markdown renders correctly
   - File sizes are reasonable (<500KB)
   - Timing is appropriate (demos ~30 min, labs ~60-90 min total)
8. **Create week-specific test script:**
   - Copy and customize test script to `exercises/week_XX_topic/test_notebooks.sh`
   - Make it executable
   - Test that it runs successfully

---

## Checklist (Use this for every notebook)

Before marking a notebook as complete:

- [ ] Week title and learning objectives clear
- [ ] Environment setup instructions included (environment-specific)
- [ ] Real-world context/storytelling for the week
- [ ] Each topic has: theory → demo → lab structure
- [ ] All demo code is heavily commented
- [ ] Lab instructions are detailed and step-by-step
- [ ] Code examples in markdown use ` ```python ``` ` blocks
- [ ] Public datasets only, with clear documentation
- [ ] Starter code uses `None # YOUR CODE` pattern
- [ ] Solution notebook has complete code with extensive comments
- [ ] Optional/extra lab included at end
- [ ] Markdown renders correctly (verified)
- [ ] Notebook runs top-to-bottom without errors
- [ ] Timing appropriate (estimate provided)
- [ ] Tone is friendly but professional
- [ ] All labs build toward week's main outcome
- [ ] File size is reasonable

---

## CRITICAL: NotebookEdit Best Practices

**⚠️ ALWAYS INSERT CELLS AFTER A SPECIFIC CELL, NEVER AT THE TOP!**

### ❌ WRONG - Will insert at top of notebook:
```python
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_type="markdown",
    edit_mode="insert",
    new_source="content"
)
```

### ✅ CORRECT - Insert after specific cell:
```python
NotebookEdit(
    notebook_path="path/to/notebook.ipynb",
    cell_id="cell-5",  # ← CRITICAL: Specify the cell to insert AFTER
    cell_type="markdown",
    edit_mode="insert",
    new_source="content"
)
```

### Workflow for Building Notebooks:

1. **Create empty notebook** with Write tool (basic JSON structure)
2. **Add first cell** with `edit_mode="insert"` (no cell_id needed for first cell)
3. **Add subsequent cells** ALWAYS using `cell_id` parameter:
   ```python
   # Add cell 1
   NotebookEdit(..., cell_id="cell-0", ...)  # Insert after cell-0

   # Add cell 2
   NotebookEdit(..., cell_id="cell-1", ...)  # Insert after cell-1

   # Add cell 3
   NotebookEdit(..., cell_id="cell-2", ...)  # Insert after cell-2
   ```

4. **Track cell IDs**: Note the returned cell_id from each NotebookEdit call and use it for the next insertion

### Common Mistake:
- **Do NOT** forget the `cell_id` parameter after the first cell
- **Do NOT** assume cells will be inserted at the end automatically
- **Always specify** which cell to insert after

---

**NOW BEGIN:**

1. Read all context files
2. Show me the plan for week [NUMBER] with environment [ENVIRONMENT]
3. Wait for my approval
4. Start building incrementally (5 cells at a time)
5. **REMEMBER**: Always use `cell_id` parameter to insert cells in correct order!
