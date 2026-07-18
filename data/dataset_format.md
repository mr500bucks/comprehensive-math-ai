# Dataset Format

The dataset trains the model to understand student reasoning rather than only generate solutions.

## Example Structure

Each example contains:

- Problem
- Student solution
- Correct solution
- Error diagnosis
- Feedback
- Hint progression


## Future Format

Example:

{
 "problem": "...",

 "student_solution": [
   "Step 1...",
   "Step 2..."
 ],

 "diagnosis": {
   "correct": false,
   "first_error_step": 2,
   "error_type": "distribution_error"
 },

 "feedback": "...",

 "hint_levels": {
   "1": "...",
   "2": "...",
   "3": "..."
 }
}
