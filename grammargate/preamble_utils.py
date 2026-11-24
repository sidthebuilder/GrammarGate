"""
Preamble utilities for GrammarGate.
Provides grammar envelope wrappers and preamble-tolerant CFG construction
to prevent instruct-tuned models from false-tripping on raw code CFGs.
"""

from __future__ import annotations


def wrap_ebnf_with_preamble(grammar_ebnf: str, allow_code_fences: bool = True) -> str:
    """
    Wraps an EBNF grammar with an optional conversational preamble or markdown code fence.
    Instruct models often emit 'Certainly! Here is the code:\n```python\n' before the raw syntax.
    By allowing an optional preamble envelope in the CFG, the model's natural conversational
    onset is grammatical, driving benign false positive rates toward 0%.

    Parameters:
        grammar_ebnf: The original EBNF grammar string.
        allow_code_fences: Whether to permit markdown ```python ... ``` fences.

    Returns:
        Augmented EBNF grammar string.
    """
    if "root ::=" not in grammar_ebnf:
        return grammar_ebnf

    # Preamble definition that permits conversational salutation and optional code block markers
    envelope_rules = r"""
preamble ::= ( [a-zA-Z0-9 ,.!:?_\-'"\t] | "\n" )*
code_open ::= "```" [a-zA-Z0-9_]* "\n"
code_close ::= "\n```" | ""
"""
    # Replace root ::= with envelope-wrapped root
    lines = grammar_ebnf.strip().splitlines()
    new_lines = []
    found_root = False
    for line in lines:
        if line.strip().startswith("root ::=") and not found_root:
            orig_expr = line.split("::=", 1)[1].strip()
            new_lines.append(f"root ::= (preamble)? (code_open)? inner_root (code_close)?")
            new_lines.append(f"inner_root ::= {orig_expr}")
            found_root = True
        else:
            new_lines.append(line)

    if found_root:
        return "\n".join(new_lines) + "\n" + envelope_rules
    return grammar_ebnf
