---
description: Create a TDD-style plan through iterative hypothesis refinement (3 cycles)
---

Create a comprehensive TDD-style plan for: $ARGUMENTS

**CRITICAL: All process and cycles must be visible IN CHAT. Do not skip steps or summarize - show full reasoning.**

## Process Overview

You will iterate through 3 complete cycles of hypothesis formation, research, refutation, and refinement. Each cycle must be fully documented in the chat.

## Cycle Structure (Repeat 3 Times)

### Cycle 1, 2, 3: [Title the cycle]

#### Step 1: Scan Relevant Context & Form Initial Hypothesis

**DO THIS IN CHAT:**
1. Identify all relevant files and code sections to read
2. Read the relevant code and understand current implementation
3. Review related documentation (CLAUDE.md, vibe files, etc.)
4. Understand the problem/feature request
5. **Formulate your initial hypothesis** - Write out:
   - What you think needs to be done
   - How you plan to approach it
   - What assumptions you're making
   - What the expected outcome should be

**Files to always check first:**
- CLAUDE.md
- vibe/*.md (all vibe files)
- Relevant app/ files based on the task
- Related tests/ if they exist

#### Step 2: Research Potential Shortcomings (Optional but Recommended)

**DO THIS IN CHAT:**
- If the approach has potential risks or unknowns, perform web searches
- Search for:
  - Best practices for similar patterns
  - Known pitfalls or issues
  - Performance considerations
  - Security concerns
  - Compatibility issues
- Document findings in chat

**If research is not needed, explicitly state why:**
- "Research not needed because: [reason]"

#### Step 3: Refute Initial Hypothesis

**DO THIS IN CHAT:**
- Critically analyze your initial hypothesis based on:
  - Code review findings
  - Research results (if performed)
  - Architecture patterns and constraints
  - Testing requirements
  - Edge cases and error scenarios
- List all potential issues:
  - What could go wrong?
  - What assumptions might be wrong?
  - What edge cases aren't covered?
  - What violates existing patterns?
  - What performance/scalability issues exist?
  - What security concerns exist?

#### Step 4: Refine Thinking & Create New Plan

**DO THIS IN CHAT:**
- Based on the refutation, create a refined plan:
  - Address the issues identified
  - Adjust the approach
  - Add missing considerations
  - Refine test strategy
  - Update assumptions
- Document the refined hypothesis

---

## After All 3 Cycles

### Final Plan Synthesis

Create the final comprehensive plan that incorporates learnings from all 3 cycles:

1. **Problem Statement**
   - Clear description of what needs to be implemented
   - Context and constraints

2. **Architecture/Design Decisions**
   - Final approach after 3 iterations
   - Rationale for decisions
   - Alternatives considered

3. **Implementation Steps (TDD Style)**
   - Step 1: Write failing test(s)
   - Step 2: Implement minimal code to pass
   - Step 3: Refactor
   - Continue test-first approach for each component

4. **Test Strategy**
   - Unit tests to write
   - Integration tests needed
   - Edge cases to cover
   - Error scenarios

5. **Files to Modify/Create**
   - Specific file paths
   - What changes in each file

6. **Success Criteria**
   - How to verify the implementation works
   - What tests must pass
   - What manual verification is needed

7. **Potential Risks & Mitigations**
   - Risks identified through iterations
   - How to mitigate them

8. **Next Steps**
   - Immediate action items
   - Dependencies to resolve first

---

## Important Notes

- **DO NOT** skip showing the process - all cycles must be visible
- **DO NOT** summarize cycles - show full reasoning each time
- **DO NOT** rush to the final plan - the iterative process is the value
- **DO** be thorough in refutation - challenge your own assumptions
- **DO** incorporate learnings from previous cycles into the next
- **DO** read actual code files, not just summaries
- **DO** consider the existing architecture patterns (see CLAUDE.md and vibe files)

