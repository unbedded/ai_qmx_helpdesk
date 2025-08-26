# CLAUDE.md — Project Memory & Coding Standards

## Role & Expectations
- You are an experienced Python developer skilled in translating requirements into **Python 3.13.5**-compatible code.  
- Implement Pythonic error handling and debugging techniques, ensuring clarity.

---

## Coding Standards
- Adhere to **PEP8** and use **type hints** consistently.
- Import and use generics from `typing` (e.g., `List`, `Dict`, `Tuple`).
- Use **named arguments** for functions with multiple parameters.
- Replace magic numbers with **constants**.

---

## Documentation
- Each file includes a header with:
  - Today’s date as `<DATE>`.
  - Explanations inside comments.
  - An `Example usage:` section in the header (not at end of file).
- For each significant step, add `STEP_ACTION_TABLE` entries in comments: `STEP_%d`.
- Provide **verbose docstrings** for all public classes, methods, and functions.

---

## Error Handling
- Use Pythonic `try-except` blocks.
- Raise built-in or custom exceptions as appropriate.
- Provide clear and informative error messages.
- Use `logging.exception()` to capture stack traces.

---

## Debugging & Logging
- Instantiate logging as the **first step in any constructor**.
- Use the `logging` module with lazy `%` formatting.
- Configure logging with file output and timestamps in the format.
- Use correct logging levels:
  - `DEBUG` for detailed information  
  - `INFO` for general events  
  - `ERROR` for exceptions  
- Default logger level: `WARNING`.
- Ensure logging is **thread-safe**.
- Log key operations (e.g., input validation, cache access, calculations).
- Avoid sensitive or redundant information.

---

## Configuration Management (Required in Every Class)
- Constructor includes parameter:  
  ```python
  def __init__(self, cfg_dict: Dict = {})
- Use a helper to initialize config params:
- If a key is missing, log a message and apply a default value.
- Implement:
    - get_cfg() → returns cfg_dict updated with current parameters.
    - set_cfg(cfg_dict) → updates matching keys:
    - Log updated parameters.
    - Log key/parameter mismatches.