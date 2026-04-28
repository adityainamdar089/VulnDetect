def check_syntax(code: str) -> tuple[bool, str]:
    """
    Basic heuristic syntax checker for code snippets.
    Checks for unbalanced brackets, parentheses, and unclosed quotes/comments.
    """
    if not code.strip():
        return True, "—"

    stack = []
    pairs = {')': '(', '}': '{', ']': '['}
    quotes = {'"': False, "'": False}
    
    in_line_comment = False
    in_block_comment = False
    
    i = 0
    while i < len(code):
        char = code[i]
        
        if in_line_comment:
            if char == '\n':
                in_line_comment = False
            i += 1
            continue
            
        if in_block_comment:
            if char == '*' and i + 1 < len(code) and code[i+1] == '/':
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue
            
        if not quotes['"'] and not quotes["'"]:
            if char == '/' and i + 1 < len(code):
                if code[i+1] == '/':
                    in_line_comment = True
                    i += 2
                    continue
                elif code[i+1] == '*':
                    in_block_comment = True
                    i += 2
                    continue
                    
            if char in "({[":
                stack.append((char, i))
            elif char in ")}]":
                if not stack:
                    return False, f"❌ Unmatched '{char}'"
                top_char, _ = stack.pop()
                if pairs[char] != top_char:
                    return False, f"❌ Mismatched '{char}' (expected '{pairs[top_char]}')"
        
        # Handle escape characters in strings
        if char == '\\' and (quotes['"'] or quotes["'"]):
            i += 2
            continue
            
        if char == '"' and not quotes["'"]:
            quotes['"'] = not quotes['"']
        elif char == "'" and not quotes['"']:
            quotes["'"] = not quotes["'"]
            
        i += 1
        
    if in_block_comment:
         return False, "❌ Unclosed block comment"
    if quotes['"']:
         return False, "❌ Unclosed double quote string"
    if quotes["'"]:
         return False, "❌ Unclosed single quote string"
    if stack:
         top_char, pos = stack.pop()
         return False, f"❌ Unclosed '{top_char}'"
         
    return True, "✅ Syntax OK"
