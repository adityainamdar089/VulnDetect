import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app import analyze_code

snippets = [
    {
        "name": "C - SQL Injection (Vulnerable)",
        "language": "C",
        "code": '''
void authenticate(char *username, char *password) {
    char query[512];
    sprintf(query, "SELECT * FROM users WHERE user='%s' AND pass='%s'", username, password);
    db_execute(query);
}
'''
    },
    {
        "name": "C - SQL Injection (Safe/Parametrized)",
        "language": "C",
        "code": '''
void authenticate(char *username, char *password) {
    db_stmt *stmt = db_prepare("SELECT * FROM users WHERE user=? AND pass=?");
    db_bind_string(stmt, 1, username);
    db_bind_string(stmt, 2, password);
    db_execute(stmt);
}
'''
    },
    {
        "name": "C++ - Buffer Overflow (Vulnerable)",
        "language": "C++",
        "code": '''
#include <iostream>
#include <cstring>

void processInput(const char* input) {
    char buffer[64];
    strcpy(buffer, input);
    std::cout << "Processing: " << buffer << std::endl;
}
'''
    },
    {
        "name": "C++ - Buffer Overflow (Safe std::string)",
        "language": "C++",
        "code": '''
#include <iostream>
#include <string>

void processInput(const std::string& input) {
    std::string buffer = input;
    std::cout << "Processing: " << buffer << std::endl;
}
'''
    },
    {
        "name": "Java - Hardcoded Credentials (Vulnerable)",
        "language": "Java",
        "code": '''
public class DbConnection {
    public void connect() {
        String dbUser = "admin";
        String dbPass = "SuperSecretPassword123!";
        Connection conn = DriverManager.getConnection("jdbc:mysql://localhost:3306/db", dbUser, dbPass);
    }
}
'''
    },
    {
        "name": "Java - Path Traversal (Vulnerable)",
        "language": "Java",
        "code": '''
import java.io.*;

public class FileHandler {
    public void readFile(String filename) throws IOException {
        File f = new File("/var/www/html/downloads/" + filename);
        FileInputStream fis = new FileInputStream(f);
        // ... read file ...
    }
}
'''
    },
    {
        "name": "C - Use After Free (Vulnerable)",
        "language": "C",
        "code": '''
#include <stdlib.h>
#include <stdio.h>

void process_data() {
    char *data = (char *)malloc(100);
    // ... do something ...
    free(data);
    
    // ... later ...
    printf("Data: %s\\n", data);
}
'''
    },
    {
        "name": "C++ - Double Free (Vulnerable)",
        "language": "C++",
        "code": '''
void deleteData(int* ptr) {
    delete ptr;
    // bug: deleting twice!
    delete ptr;
}
'''
    },
    {
        "name": "C++ - Memory Safe (Smart Pointers)",
        "language": "C++",
        "code": '''
#include <memory>
#include <iostream>

void processSafe() {
    std::unique_ptr<int> ptr = std::make_unique<int>(42);
    std::cout << "Value: " << *ptr << std::endl;
    // Memory automatically freed here
}
'''
    },
    {
        "name": "Java - OS Command Injection (Vulnerable)",
        "language": "Java",
        "code": '''
import java.io.*;

public class NetworkPing {
    public void pingHost(String ipAddress) throws IOException {
        String command = "ping -c 4 " + ipAddress;
        Process process = Runtime.getRuntime().exec(command);
    }
}
'''
    },
    {
        "name": "Java - OS Command Safe (ProcessBuilder)",
        "language": "Java",
        "code": '''
import java.io.*;
import java.util.*;

public class NetworkPing {
    public void pingHost(String ipAddress) throws IOException {
        // Safe: separates executable from arguments
        ProcessBuilder pb = new ProcessBuilder("ping", "-c", "4", ipAddress);
        Process process = pb.start();
    }
}
'''
    },
    {
        "name": "C - Format String Vulnerability",
        "language": "C",
        "code": '''
#include <stdio.h>

void log_message(char *user_input) {
    // Vulnerable: user input used directly as format string
    printf(user_input); 
}
'''
    },
    {
        "name": "C - Format String Safe",
        "language": "C",
        "code": '''
#include <stdio.h>

void log_message(char *user_input) {
    // Safe: format string specified
    printf("%s", user_input); 
}
'''
    },
    {
        "name": "Java - Null Pointer Dereference (Vulnerable)",
        "language": "Java",
        "code": '''
public class UserHandler {
    public void printUserName(User user) {
        // Missing null check!
        System.out.println("User is: " + user.getName().toUpperCase());
    }
}
'''
    },
    {
        "name": "C - Integer Overflow leading to BOF",
        "language": "C",
        "code": '''
#include <stdlib.h>
#include <string.h>

void process_items(unsigned short num_items) {
    // If num_items is large (e.g., 65535), num_items * sizeof(int) might overflow
    // and allocate a very small buffer!
    int *buffer = (int *)malloc(num_items * sizeof(int));
    if (buffer) {
        for(int i = 0; i < num_items; i++) {
            buffer[i] = i; // Buffer overflow!
        }
        free(buffer);
    }
}
'''
    }
]

results = []
for idx, s in enumerate(snippets):
    print(f"[{idx+1}/{len(snippets)}] Analyzing: {s['name']} ...")
    # analyze_code returns: (vulnerable, vuln_type, severity, confidence, affected_lines, llm_reasoning)
    try:
        vuln, vtype, sev, conf, lines, reasoning = analyze_code(s['code'], s['language'])
        results.append({
            "name": s['name'],
            "language": s['language'],
            "is_vulnerable": vuln,
            "type": vtype,
            "severity": sev,
            "confidence": conf,
            "affected_lines": lines,
            "reasoning": reasoning
        })
    except Exception as e:
        results.append({
            "name": s['name'],
            "error": str(e)
        })

with open("test_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("Done. Results saved to test_results.json")
