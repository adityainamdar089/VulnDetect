import re

def taint_depth_score(code: str, language: str) -> float:
    sources_pattern = r"(argv|stdin|gets|scanf|input\(|request\.GET|request\.POST|request\.args|System\.in|Scanner)"
    sanitizers_pattern = r"(\?|prepareStatement|htmlspecialchars|html\.escape|bleach\.clean|validate|sanitize)"
    sources = len(re.findall(sources_pattern, code, re.IGNORECASE))
    sanitizers = len(re.findall(sanitizers_pattern, code, re.IGNORECASE))
    return float(min(sources / (sanitizers + 1), 1.0))

def api_danger_score(code: str, language: str) -> float:
    lang = language.lower()
    danger_count = 0
    if "c" in lang:
        danger_count += len(re.findall(r"\b(strcpy|strcat|gets|sprintf|vsprintf|memcpy)\b", code))
    elif "py" in lang:
        danger_count += len(re.findall(r"\b(eval|exec|pickle\.load|subprocess\.call.*shell=True|os\.system)\b", code))
    elif "java" in lang:
        danger_count += len(re.findall(r"\b(Runtime\.exec|ProcessBuilder|ObjectInputStream|Class\.forName)\b", code))
        
    total_calls = len(re.findall(r"\b\w+\s*\(", code))
    return float(min(danger_count / (total_calls + 1), 1.0))

def allocation_safety_score(code: str, language: str) -> float:
    lang = language.lower()
    if "c" not in lang:
        return 0.1
        
    allocs = len(re.findall(r"\b(malloc|calloc|realloc|new)\b", code))
    frees = len(re.findall(r"\b(free|delete)\b", code))
    return float(1.0 - min(frees / (allocs + 1.0), 1.0))

def control_flow_danger_score(code: str, language: str) -> float:
    patterns = 0
    lines = code.split('\n')
    for i, line in enumerate(lines):
        if "malloc" in line:
            check_found = False
            for j in range(i, min(i+4, len(lines))):
                if re.search(r"if.*(?i:null)", lines[j]):
                    check_found = True
                    break
            if not check_found:
                patterns += 1
                
    for i, line in enumerate(lines):
        if re.search(r"=\s*\w+\(", line):
            check_found = False
            for j in range(i, min(i+3, len(lines))):
                if "if " in lines[j] or "if(" in lines[j]:
                    check_found = True
                    break
            if not check_found:
                patterns += 1
                
    for i, line in enumerate(lines):
        if re.search(r"\b(for|while)\b", line) and "[" in line and "]" in line:
            if not re.search(r"(<|>|length|size)", line):
                patterns += 1

    return float(min(patterns / 5.0, 1.0))

def injection_surface_score(code: str, language: str) -> float:
    matches = 0
    matches += len(re.findall(r"(query.*\+|sql.*\+|f\".*(SELECT|INSERT|UPDATE|DELETE).*\")", code, re.IGNORECASE))
    matches += len(re.findall(r"(exec.*\+|system.*\+|os\.system.*%)", code, re.IGNORECASE))
    matches += len(re.findall(r"(.*\/.*\+|os\.path\.join.*)", code, re.IGNORECASE))
    return float(min(matches / 5.0, 1.0))

def credential_exposure_score(code: str, language: str) -> float:
    matches = len(re.findall(r"(password|passwd|secret|api_key|token|apikey)\s*=\s*[\"'][^\"']{4,}[\"']", code, re.IGNORECASE))
    return float(min(matches / 3.0, 1.0))

def extract_security_features(code: str, language: str) -> dict:
    return {
        "taint_depth_score": taint_depth_score(code, language),
        "api_danger_score": api_danger_score(code, language),
        "allocation_safety_score": allocation_safety_score(code, language),
        "control_flow_danger_score": control_flow_danger_score(code, language),
        "injection_surface_score": injection_surface_score(code, language),
        "credential_exposure_score": credential_exposure_score(code, language)
    }

SECURITY_FEATURE_COLS = [
    "taint_depth_score", 
    "api_danger_score", 
    "allocation_safety_score", 
    "control_flow_danger_score", 
    "injection_surface_score", 
    "credential_exposure_score"
]
