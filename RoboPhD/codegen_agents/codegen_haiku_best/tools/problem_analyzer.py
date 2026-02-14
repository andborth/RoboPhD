#!/usr/bin/env python3
"""
Reflection-Refined Problem Analyzer for Code Generation Critic (Iteration 5)

Based on iter2_deep_analysis_critic with improvements from reflections:
- iter2: Full AST complexity analysis, constraint extraction, TLE risk, pattern detection
- iter3: Conservative sample parsing (fail safe on uncertain formats)
- iter4: LaTeX/Unicode constraint normalization, improved sample parsing patterns

This tool analyzes Python solutions for competitive programming problems,
providing structured feedback on:
1. Time complexity analysis
2. Code pattern detection (common bugs)
3. Sample test execution
4. Constraint validation
"""

import ast
import re
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any


class ComplexityAnalyzer(ast.NodeVisitor):
    """Analyzes code complexity by examining loop structures and operations."""

    def __init__(self):
        self.loop_depth = 0
        self.max_loop_depth = 0
        self.loops = []
        self.current_loop_stack = []
        self.function_calls = []
        self.potential_issues = []
        self.recursive_functions = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Detect recursive functions using AST (from iter3/iter4)."""
        func_name = node.name
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == func_name:
                    self.recursive_functions.append({
                        "name": func_name,
                        "line": node.lineno,
                    })
                    break
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)

        loop_info = self._analyze_loop(node)
        nested_in = list(self.current_loop_stack)
        self.current_loop_stack.append(loop_info)
        self.loops.append({
            "type": "for",
            "depth": self.loop_depth,
            "line": node.lineno,
            "info": loop_info,
            "nested_in": nested_in,
        })

        self.generic_visit(node)

        self.current_loop_stack.pop()
        self.loop_depth -= 1

    def visit_While(self, node: ast.While) -> None:
        self.loop_depth += 1
        self.max_loop_depth = max(self.max_loop_depth, self.loop_depth)

        self.current_loop_stack.append("while loop")
        self.loops.append({
            "type": "while",
            "depth": self.loop_depth,
            "line": node.lineno,
            "info": "while loop (unbounded)",
            "nested_in": list(self.current_loop_stack[:-1]),
        })

        self.generic_visit(node)

        self.current_loop_stack.pop()
        self.loop_depth -= 1

    def visit_ListComp(self, node: ast.ListComp) -> None:
        num_generators = len(node.generators)
        if num_generators >= 2:
            self.potential_issues.append({
                "type": "nested_comprehension",
                "line": node.lineno,
                "detail": f"List comprehension with {num_generators} nested generators",
            })
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        func_name = self._get_func_name(node)
        if func_name:
            self.function_calls.append({
                "name": func_name,
                "line": node.lineno,
                "in_loop_depth": self.loop_depth,
            })

            # Full list of expensive operations (from iter2 - keep ALL)
            expensive_ops = {
                "sorted": "O(n log n)",
                "sort": "O(n log n)",
                "list": "O(n) copy",
                "set": "O(n) conversion",
                "dict": "O(n) conversion",
                "sum": "O(n)",
                "max": "O(n)",
                "min": "O(n)",
                "count": "O(n)",
                "index": "O(n)",
                "remove": "O(n)",
                "insert": "O(n)",
                "pop": "O(n) for non-last element",
            }

            if func_name in expensive_ops and self.loop_depth > 0:
                self.potential_issues.append({
                    "type": "expensive_in_loop",
                    "line": node.lineno,
                    "detail": f"`{func_name}` ({expensive_ops[func_name]}) called inside {self.loop_depth}-deep loop",
                })

        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        for op in node.ops:
            if isinstance(op, ast.In) and self.loop_depth > 0:
                self.potential_issues.append({
                    "type": "membership_in_loop",
                    "line": node.lineno,
                    "detail": "`in` operator inside loop - O(n) if checking list",
                })
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.slice, ast.Slice) and self.loop_depth > 0:
            self.potential_issues.append({
                "type": "slice_in_loop",
                "line": node.lineno,
                "detail": "List/string slicing inside loop creates copies (O(n) each)",
            })
        self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        if isinstance(node.op, ast.Add) and self.loop_depth > 0:
            self.potential_issues.append({
                "type": "potential_string_concat",
                "line": node.lineno,
                "detail": "Addition in loop - if string concatenation, this is O(n) per operation",
            })
        self.generic_visit(node)

    def _analyze_loop(self, node: ast.For) -> str:
        """Extract information about loop bounds."""
        if isinstance(node.iter, ast.Call):
            func_name = self._get_func_name(node.iter)
            if func_name == "range":
                args = node.iter.args
                if len(args) == 1:
                    return f"range({self._expr_to_str(args[0])})"
                elif len(args) == 2:
                    return f"range({self._expr_to_str(args[0])}, {self._expr_to_str(args[1])})"
                elif len(args) == 3:
                    return f"range({self._expr_to_str(args[0])}, {self._expr_to_str(args[1])}, {self._expr_to_str(args[2])})"
            return f"iterating over {func_name}(...)"
        elif isinstance(node.iter, ast.Name):
            return f"iterating over {node.iter.id}"
        elif isinstance(node.iter, ast.Subscript):
            return "iterating over subscript"
        return "for loop"

    def _get_func_name(self, node: ast.Call) -> str | None:
        """Extract function name from a Call node."""
        if isinstance(node.func, ast.Name):
            return node.func.id
        elif isinstance(node.func, ast.Attribute):
            return node.func.attr
        return None

    def _expr_to_str(self, node: ast.expr) -> str:
        """Convert an expression node to a readable string."""
        if isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.BinOp):
            left = self._expr_to_str(node.left)
            right = self._expr_to_str(node.right)
            op = self._binop_to_str(node.op)
            return f"({left} {op} {right})"
        elif isinstance(node, ast.Call):
            func = self._get_func_name(node)
            return f"{func}(...)"
        elif isinstance(node, ast.Subscript):
            return f"{self._expr_to_str(node.value)}[...]"
        return "?"

    def _binop_to_str(self, op: ast.operator) -> str:
        """Convert binary operator to string."""
        ops = {
            ast.Add: "+",
            ast.Sub: "-",
            ast.Mult: "*",
            ast.Div: "/",
            ast.FloorDiv: "//",
            ast.Mod: "%",
            ast.Pow: "**",
        }
        return ops.get(type(op), "?")


def analyze_complexity(code: str) -> dict[str, Any]:
    """Analyze the time complexity of the given code."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return {"error": f"Syntax error: {e}"}

    analyzer = ComplexityAnalyzer()
    analyzer.visit(tree)

    return {
        "max_loop_depth": analyzer.max_loop_depth,
        "loops": analyzer.loops,
        "potential_issues": analyzer.potential_issues,
        "expensive_calls_in_loops": [
            c for c in analyzer.function_calls if c["in_loop_depth"] > 0
        ],
        "recursive_functions": analyzer.recursive_functions,
    }


def normalize_constraint_text(text: str) -> str:
    """Normalize constraint notation to standard format.

    Handles:
    - LaTeX escapes: \\leq, \\times, \\cdot
    - Unicode superscripts: ¹²³⁴⁵⁶⁷⁸⁹⁰
    - Various multiplication symbols
    """
    if not text:
        return ""

    # Handle LaTeX escapes
    text = text.replace("\\leq", "<=")
    text = text.replace("\\le", "<=")
    text = text.replace("\\geq", ">=")
    text = text.replace("\\ge", ">=")
    text = text.replace("\\times", "×")
    text = text.replace("\\cdot", "×")
    text = text.replace("\\,", "")  # Remove thin spaces

    # Handle Unicode superscripts -> ^digit format
    superscripts = {
        "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
        "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9"
    }
    for sup, digit in superscripts.items():
        text = text.replace(sup, "^" + digit)

    # Normalize multiplication symbols
    text = text.replace("*", "×")
    text = text.replace("x10", "×10")  # lowercase x before 10

    return text


def extract_constraints(reflection: str, problem: str = "") -> dict[str, Any]:
    """Extract constraint values from reflection and problem text.

    Handles multiple formats:
    - Plain numbers: N <= 100000
    - Scientific: N <= 10^5, N ≤ 2×10^5, N <= 2*10^5
    - LaTeX: N \\leq 2 \\times 10^5
    - Unicode: N ≤ 2×10⁵
    """
    constraints = {}

    # Normalize texts
    problem_norm = normalize_constraint_text(problem)
    reflection_norm = normalize_constraint_text(reflection)

    # Prioritize the Constraints section from problem.md if available
    constraint_section = ""
    for section_name in ["Constraints", "Constraint", "制約"]:  # Include Japanese
        if section_name in problem_norm:
            parts = problem_norm.split(section_name)
            if len(parts) > 1:
                # Get until next section or blank lines
                constraint_section = parts[1].split("\n\n")[0]
                break

    primary_text = constraint_section if constraint_section else problem_norm
    fallback_text = reflection_norm

    def parse_value(val_str: str) -> int | None:
        """Parse a constraint value, handling various formats."""
        val_str = val_str.strip().replace(",", "").replace(" ", "")

        # Check for "10^X" format
        exp_match = re.match(r"10\^(\d+)", val_str)
        if exp_match:
            return 10 ** int(exp_match.group(1))

        # Check for "A×10^X" format (various multiplication symbols)
        sci_match = re.match(r"(\d+)\s*[×x*]\s*10\^?(\d+)", val_str)
        if sci_match:
            return int(sci_match.group(1)) * (10 ** int(sci_match.group(2)))

        # Plain number
        try:
            return int(val_str)
        except ValueError:
            return None

    # Value pattern - match constraint values
    value_pattern = r"(\d+\s*[×x*]\s*10\s*\^\s*\d+|10\s*\^\s*\d+|\d{4,})"

    # Patterns for constraint extraction
    inequality_patterns = [
        # Array/string length patterns
        (r"(?:nums|array|s|str|string)\.?(?:length)?\s*[≤<]=?\s*" + value_pattern, "n"),
        # Standard variable patterns
        (r"\b[Nn]\s*[≤<]=?\s*" + value_pattern, "n"),
        (r"\b[Mm]\s*[≤<]=?\s*" + value_pattern, "m"),
        (r"\b[Qq]\s*[≤<]=?\s*" + value_pattern, "q"),
        (r"\b[Kk]\s*[≤<]=?\s*" + value_pattern, "k"),
        (r"\b[Tt]\s*[≤<]=?\s*" + value_pattern, "t"),  # Test cases
        # Combined N, M patterns
        (r"[Nn],?\s*[Mm]\s*[≤<]=?\s*" + value_pattern, "n_and_m"),
        # Sum of N pattern (important for multi-test problems)
        (r"sum\s+of\s+[Nn]\s*[≤<]=?\s*" + value_pattern, "sum_n"),
    ]

    for text in [primary_text, problem_norm, fallback_text]:
        if not text:
            continue

        for pattern, name in inequality_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                value = parse_value(match)
                if value and value >= 100:  # Ignore tiny values
                    if name == "n_and_m":
                        if "n" not in constraints or value > constraints.get("n", 0):
                            constraints["n"] = value
                        if "m" not in constraints or value > constraints.get("m", 0):
                            constraints["m"] = value
                    else:
                        if name not in constraints or value > constraints.get(name, 0):
                            constraints[name] = value

        if constraints:
            break

    return constraints


def estimate_operations(complexity: dict, constraints: dict) -> dict[str, Any]:
    """Estimate number of operations based on complexity and constraints."""
    result = {
        "estimated_ops": None,
        "risk_level": "unknown",
        "explanation": "",
    }

    max_depth = complexity.get("max_loop_depth", 0)
    n = constraints.get("n", constraints.get("n_and_m", constraints.get("generic", 0)))
    m = constraints.get("m", constraints.get("n_and_m", n))
    q = constraints.get("q", 1)

    if n == 0:
        result["explanation"] = "Could not determine constraints from problem"
        return result

    loops = complexity.get("loops", [])
    has_q_outer = any("Q" in str(l.get("info", "")) or "q" in str(l.get("info", "")).lower() for l in loops)

    # Estimate based on loop depth and structure
    if max_depth == 1:
        ops = max(n, q)
        result["complexity_estimate"] = f"O(n) or O(q) ≈ {ops:,.0f}"
    elif max_depth == 2:
        if q > 1 and has_q_outer:
            ops = n * q
            result["complexity_estimate"] = f"O(n*q) ≈ {ops:,.0f}"
        elif m != n and m > 1:
            ops = n * m
            result["complexity_estimate"] = f"O(n*m) ≈ {ops:,.0f}"
        else:
            ops = n * n
            result["complexity_estimate"] = f"O(n²) ≈ {ops:,.0f}"
    elif max_depth == 3:
        ops = n * n * n
        result["complexity_estimate"] = f"O(n³) ≈ {ops:,.0f}"
    else:
        ops = n ** max_depth
        result["complexity_estimate"] = f"O(n^{max_depth}) ≈ {ops:,.0f}"

    result["estimated_ops"] = ops

    # Risk assessment with calibrated thresholds
    if ops > 10**10:
        result["risk_level"] = "CRITICAL"
        result["explanation"] = f"~{ops:,.0f} operations will almost certainly TLE (>100 seconds)"
    elif ops > 5 * 10**9:
        result["risk_level"] = "HIGH"
        result["explanation"] = f"~{ops:,.0f} operations is very risky (likely 50-100+ seconds)"
    elif ops > 10**9:
        result["risk_level"] = "MODERATE-HIGH"
        result["explanation"] = f"~{ops:,.0f} operations is risky (may TLE depending on operations)"
    elif ops > 10**8:
        result["risk_level"] = "MODERATE"
        result["explanation"] = f"~{ops:,.0f} operations may be tight (1-10 seconds estimated)"
    else:
        result["risk_level"] = "LOW"
        result["explanation"] = f"~{ops:,.0f} operations should be safe (<1 second estimated)"

    return result


def detect_code_patterns(code: str, complexity: dict) -> list[dict[str, str]]:
    """Detect common bug patterns in the code.

    Keeps ALL detections (no filtering per iter2/iter4 lessons).
    """
    patterns = []

    # Check for recursion issues (using AST results)
    recursive_funcs = complexity.get("recursive_functions", [])
    for func in recursive_funcs:
        if "@lru_cache" not in code and "@cache" not in code and "memo" not in code.lower():
            patterns.append({
                "type": "recursion_no_memo",
                "detail": f"Recursive function `{func['name']}` (line {func['line']}) without memoization - may cause TLE",
            })
        if "setrecursionlimit" not in code:
            patterns.append({
                "type": "recursion_limit",
                "detail": f"Recursive function `{func['name']}` without sys.setrecursionlimit - default 1000 may cause RecursionError",
            })

    # Pattern: Using list for membership when set would be better
    if re.search(r"if\s+\w+\s+in\s+\w+\s*:", code):
        patterns.append({
            "type": "membership_test",
            "detail": "Uses `in` operator - ensure collection is a set/dict if checked repeatedly",
        })

    # Pattern: Reading input in a loop
    if re.search(r"for.*:.*\n.*input\(\)", code, re.MULTILINE):
        patterns.append({
            "type": "input_in_loop",
            "detail": "Reading input inside loop - consider reading all input first",
        })

    # Pattern: String concatenation in loop
    if re.search(r"for.*:.*\n.*\+\s*=.*[\"']", code, re.MULTILINE):
        patterns.append({
            "type": "string_concat_loop",
            "detail": "String concatenation in loop - use list.append() and ''.join() instead",
        })

    # Pattern: Modifying list while iterating
    if re.search(r"for\s+\w+\s+in\s+(\w+).*:.*\n.*\1\.(append|remove|pop|insert)", code, re.MULTILINE):
        patterns.append({
            "type": "modify_during_iteration",
            "detail": "Potentially modifying list while iterating - may cause unexpected behavior",
        })

    # Pattern: Division without handling zero
    if re.search(r"/[^/]", code) and "ZeroDivision" not in code and "== 0" not in code and "!= 0" not in code:
        patterns.append({
            "type": "possible_division_by_zero",
            "detail": "Division operation without apparent zero check",
        })

    # Pattern: Array access with offset
    if re.search(r"\[\s*\w+\s*[\+\-]\s*\d+\s*\]", code):
        patterns.append({
            "type": "offset_array_access",
            "detail": "Array access with offset (e.g., arr[i-1], arr[i+1]) - ensure bounds are valid",
        })

    # Pattern: Float comparison
    if re.search(r"==\s*\d+\.\d+|==.*float|float.*==", code):
        patterns.append({
            "type": "float_equality",
            "detail": "Float equality comparison - may have precision issues",
        })

    # Pattern: Large powers without modulo
    if re.search(r"\*\*\s*\d{2,}", code) and "%" not in code and "mod" not in code.lower():
        patterns.append({
            "type": "large_power_no_mod",
            "detail": "Large exponentiation without modulo - may cause overflow or be slow",
        })

    return patterns


def parse_sample_tests(problem_text: str) -> list[tuple[str, str]]:
    """Parse sample test cases from problem.md with conservative approach.

    Following iter3's lesson: "Conservative parsing is better than incorrect parsing.
    When uncertain about sample format, output 'Could not parse' rather than risk
    incorrect parsing."
    """
    samples = []

    # Pattern 1: AtCoder format - "Sample Input X\n\n<input>\n\nSample Output X\n\n<output>"
    pattern1 = r"Sample Input (\d+)\s*\n\n(.*?)\n\nSample Output \1\s*\n\n(.*?)(?=\n\n(?:Sample Input|\Z)|$)"
    matches = re.findall(pattern1, problem_text, re.DOTALL)
    for num, inp, out in matches:
        out_lines = []
        for line in out.split('\n'):
            line_stripped = line.strip()
            # Stop at explanatory text (starts with letter and has multiple words)
            if line_stripped and line_stripped[0].isalpha() and ' ' in line_stripped:
                # Check if this looks like explanation vs actual output
                if not line_stripped.replace(' ', '').replace('-', '').replace('.', '').isdigit():
                    break
            out_lines.append(line)
        out_clean = '\n'.join(out_lines).strip()
        if inp.strip() and out_clean:
            samples.append((inp.strip(), out_clean))

    # Pattern 2: Code blocks format
    if not samples:
        pattern2 = r"(?:Sample\s+Input|Input|Example)\s*(?:\d+)?[:\s]*\n```[^\n]*\n(.*?)```.*?(?:Sample\s+Output|Output|Example)\s*(?:\d+)?[:\s]*\n```[^\n]*\n(.*?)```"
        samples = re.findall(pattern2, problem_text, re.DOTALL | re.IGNORECASE)

    # Pattern 3: Generic Input/Output format (more conservative - single line output only)
    if not samples:
        pattern3 = r"Input[:\s]*\n(.*?)\nOutput[:\s]*\n([^\n]+)"
        matches = re.findall(pattern3, problem_text, re.DOTALL)
        for inp, out in matches:
            # Only accept if output looks like a simple value (not explanation)
            out_clean = out.strip()
            if inp.strip() and out_clean and not out_clean.startswith("The ") and not out_clean.startswith("In "):
                samples.append((inp.strip(), out_clean))

    # Clean up and validate samples
    valid_samples = []
    for inp, out in samples:
        inp = inp.strip()
        out = out.strip()
        # Skip if output looks like an explanation
        if out and inp and not any(out.lower().startswith(x) for x in ["the ", "in ", "for ", "we ", "this "]):
            valid_samples.append((inp, out))

    return valid_samples


def normalize_output(output: str) -> str:
    """Normalize output for comparison (tolerant of whitespace differences)."""
    lines = [line.strip() for line in output.strip().split('\n')]
    lines = [line for line in lines if line]
    return '\n'.join(lines)


def run_sample_tests(code: str, problem_path: Path) -> dict[str, Any]:
    """Run the code on sample test cases."""
    results = {
        "tested": False,
        "samples_found": 0,
        "passed": 0,
        "failed": 0,
        "details": [],
    }

    if not problem_path.exists():
        results["error"] = "problem.md not found"
        return results

    problem_text = problem_path.read_text()
    samples = parse_sample_tests(problem_text)

    results["samples_found"] = len(samples)

    if not samples:
        results["note"] = "Could not parse sample test cases from problem"
        return results

    results["tested"] = True

    for i, (inp, expected) in enumerate(samples[:3]):  # Test up to 3 samples
        try:
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
                f.write(code)
                temp_path = f.name

            result = subprocess.run(
                ["python3", temp_path],
                input=inp,
                capture_output=True,
                text=True,
                timeout=5,
            )

            actual = result.stdout.strip()
            stderr = result.stderr.strip()

            if result.returncode != 0:
                results["failed"] += 1
                results["details"].append({
                    "sample": i + 1,
                    "status": "ERROR",
                    "error": stderr[:200] if stderr else "Non-zero exit code",
                })
            else:
                # Use tolerant comparison
                norm_actual = normalize_output(actual)
                norm_expected = normalize_output(expected)

                if norm_actual == norm_expected:
                    results["passed"] += 1
                    results["details"].append({
                        "sample": i + 1,
                        "status": "PASS",
                    })
                else:
                    results["failed"] += 1
                    results["details"].append({
                        "sample": i + 1,
                        "status": "WRONG_ANSWER",
                        "expected": expected[:100],
                        "actual": actual[:100],
                    })

        except subprocess.TimeoutExpired:
            results["failed"] += 1
            results["details"].append({
                "sample": i + 1,
                "status": "TIMEOUT",
                "detail": "Exceeded 5 second limit on sample input",
            })
        except Exception as e:
            results["details"].append({
                "sample": i + 1,
                "status": "ERROR",
                "error": str(e)[:100],
            })
        finally:
            try:
                Path(temp_path).unlink()
            except:
                pass

    return results


def analyze_reflection_claims(reflection: str, complexity: dict) -> list[dict[str, str]]:
    """Check if reflection claims match code analysis."""
    warnings = []

    # Check complexity claims
    if "O(n)" in reflection and complexity["max_loop_depth"] >= 2:
        warnings.append({
            "type": "complexity_mismatch",
            "detail": f"Reflection claims O(n) but code has {complexity['max_loop_depth']}-deep nested loops",
        })

    if "efficient" in reflection.lower() and complexity["max_loop_depth"] >= 3:
        warnings.append({
            "type": "efficiency_claim",
            "detail": f"Reflection claims solution is efficient but has {complexity['max_loop_depth']}-deep nesting",
        })

    # Check for overconfident claims
    overconfident_phrases = [
        "easily runs within",
        "well within the time limit",
        "this is efficient enough",
        "acceptable for the constraints",
        "no issues",
        "handles all edge cases",
        "easily meets",
        "should pass",
        "will pass",
    ]
    for phrase in overconfident_phrases:
        if phrase.lower() in reflection.lower():
            warnings.append({
                "type": "overconfident_claim",
                "detail": f"Reflection contains '{phrase}' - verify independently",
            })

    return warnings


def main():
    """Main entry point."""
    output_dir = Path("tool_output")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "critic_feedback.txt"

    solution_path = Path("solution.py")
    reflection_path = Path("reflection.md")
    problem_path = Path("problem.md")

    output_lines = []
    output_lines.append("# CRITIC ANALYSIS REPORT\n")

    # Read inputs
    if not solution_path.exists():
        output_lines.append("ERROR: solution.py not found")
        output_path.write_text("\n".join(output_lines))
        return 1

    code = solution_path.read_text()
    reflection = reflection_path.read_text() if reflection_path.exists() else ""
    problem = problem_path.read_text() if problem_path.exists() else ""

    # Track critical issues for summary
    critical_issues = []

    # 1. Complexity Analysis
    output_lines.append("## 1. COMPLEXITY ANALYSIS\n")
    complexity = analyze_complexity(code)

    if "error" in complexity:
        output_lines.append(f"Parse error: {complexity['error']}\n")
    else:
        output_lines.append(f"**Maximum loop nesting depth**: {complexity['max_loop_depth']}\n")

        if complexity["loops"]:
            output_lines.append("**Loop structure:**")
            for loop in complexity["loops"]:
                indent = "  " * loop["depth"]
                output_lines.append(f"{indent}- Line {loop['line']}: {loop['info']}")
            output_lines.append("")

        if complexity["potential_issues"]:
            output_lines.append("**Potential performance issues:**")
            for issue in complexity["potential_issues"]:
                output_lines.append(f"- Line {issue['line']}: {issue['detail']}")
            output_lines.append("")

    # 2. Constraint Analysis & TLE Risk
    output_lines.append("## 2. CONSTRAINT & TLE RISK ASSESSMENT\n")
    constraints = extract_constraints(reflection, problem)

    ops_estimate = {}
    if constraints:
        output_lines.append("**Detected constraints:**")
        for name, value in constraints.items():
            output_lines.append(f"- {name}: {value:,}")
        output_lines.append("")

        ops_estimate = estimate_operations(complexity, constraints)
        if ops_estimate["estimated_ops"]:
            output_lines.append(f"**Complexity estimate**: {ops_estimate.get('complexity_estimate', 'N/A')}")
            output_lines.append(f"**TLE Risk**: {ops_estimate['risk_level']}")
            output_lines.append(f"**Assessment**: {ops_estimate['explanation']}")
            output_lines.append("")

            if ops_estimate["risk_level"] in ["CRITICAL", "HIGH"]:
                critical_issues.append(f"TLE RISK: {ops_estimate['risk_level']} - {ops_estimate['explanation']}")
    else:
        output_lines.append("Could not extract constraints from reflection/problem.\n")
        # Add warning if nested loops detected without constraints
        max_depth = complexity.get("max_loop_depth", 0)
        if max_depth >= 2:
            output_lines.append(f"**WARNING**: Code has {max_depth}-deep nested loops but constraints could not be extracted.")
            output_lines.append("**YOU MUST manually check constraints from the problem statement.**")
            output_lines.append("If N ≥ 10^4-10^5 with O(N²), this is almost certain TLE!")
            output_lines.append("")
            critical_issues.append(f"POTENTIAL TLE: {max_depth}-deep nested loops detected - VERIFY CONSTRAINTS MANUALLY")

    # 3. Code Pattern Detection
    output_lines.append("## 3. CODE PATTERN ANALYSIS\n")
    patterns = detect_code_patterns(code, complexity)

    if patterns:
        output_lines.append("**Detected patterns:**")
        for p in patterns:
            output_lines.append(f"- [{p['type']}] {p['detail']}")
        output_lines.append("")
    else:
        output_lines.append("No concerning patterns detected.\n")

    # 4. Reflection Validation
    output_lines.append("## 4. REFLECTION VALIDATION\n")
    warnings = analyze_reflection_claims(reflection, complexity)

    if warnings:
        output_lines.append("**Warnings about reflection claims:**")
        for w in warnings:
            output_lines.append(f"- [{w['type']}] {w['detail']}")
        output_lines.append("")
    else:
        output_lines.append("No contradictions found between reflection and code analysis.\n")

    # 5. Sample Test Execution
    output_lines.append("## 5. SAMPLE TEST EXECUTION\n")
    test_results = run_sample_tests(code, problem_path)

    if test_results["tested"]:
        output_lines.append(f"**Samples found**: {test_results['samples_found']}")
        output_lines.append(f"**Passed**: {test_results['passed']}/{len(test_results['details'])}")

        for detail in test_results["details"]:
            status = detail["status"]
            sample_num = detail["sample"]
            if status == "PASS":
                output_lines.append(f"- Sample {sample_num}: PASS")
            elif status == "WRONG_ANSWER":
                output_lines.append(f"- Sample {sample_num}: WRONG_ANSWER")
                output_lines.append(f"  Expected: {detail.get('expected', 'N/A')}")
                output_lines.append(f"  Actual: {detail.get('actual', 'N/A')}")
                critical_issues.append(f"SAMPLE TEST FAILED: Sample {sample_num} returned wrong answer")
            elif status == "TIMEOUT":
                output_lines.append(f"- Sample {sample_num}: TIMEOUT (>5s)")
                critical_issues.append(f"SAMPLE TEST TIMEOUT: Sample {sample_num} exceeded 5 seconds")
            else:
                output_lines.append(f"- Sample {sample_num}: ERROR - {detail.get('error', 'Unknown')}")
                critical_issues.append(f"SAMPLE TEST ERROR: Sample {sample_num} - {detail.get('error', 'Unknown')}")
        output_lines.append("")
    else:
        output_lines.append(f"Note: {test_results.get('note', test_results.get('error', 'Could not run tests'))}\n")

    # 6. Critical Issues Summary
    output_lines.append("## 6. CRITICAL ISSUES SUMMARY\n")

    if critical_issues:
        output_lines.append("**THE FOLLOWING CRITICAL ISSUES WERE DETECTED:**\n")
        for i, issue in enumerate(critical_issues, 1):
            output_lines.append(f"{i}. {issue}")
        output_lines.append("")
        output_lines.append("These issues strongly indicate the code will fail on hidden test cases.")
        output_lines.append("**Default to INCORRECT unless you can prove the analysis is wrong.**")
    else:
        output_lines.append("No critical issues detected by automated analysis.\n")
        output_lines.append("**However**, automated analysis cannot catch all bugs.")
        output_lines.append("Before saying CORRECT, you MUST:")
        output_lines.append("1. Verify algorithm correctness for ALL valid inputs (not just samples)")
        output_lines.append("2. Check edge cases: empty input, single element, boundary values")
        output_lines.append("3. TRY TO CONSTRUCT A COUNTEREXAMPLE that breaks the algorithm")
        output_lines.append("4. Verify all conditional branches handle their cases correctly")
        output_lines.append("")
        output_lines.append("If you cannot verify all of these, say INCORRECT.")

    output_lines.append("")

    # Write output
    output = "\n".join(output_lines)
    output_path.write_text(output)

    print(f"Analysis complete - wrote {len(output)} bytes to {output_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        # Write error to output file for debugging
        output_dir = Path("tool_output")
        output_dir.mkdir(exist_ok=True)
        error_output = f"# ANALYSIS ERROR\n\nTool encountered an error:\n```\n{traceback.format_exc()}\n```\n"
        (output_dir / "critic_feedback.txt").write_text(error_output)
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
