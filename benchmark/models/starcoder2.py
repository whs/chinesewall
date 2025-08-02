from typing import List

from .chat_base import Message


def starcoder2_edit_prompt_1shot(old: str, instr: str, _: str) -> str:
    return f"""<issue_start>username_0: I have a program in Python that I'd like to change.

Here is the code for the program:
```py
def add(a, b):
    return a + b
```

The change I'd like to make is:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.

Please someone help me. Can you also provide the full code with the change?<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

    def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```
Upvotes: 200<issue_comment>username_0: Thank you so much! I have another program in Python that I'd like to change.

Here is the code for the program:
```py
{old}
```

The change I'd like to make is:
{instr}

Please someone help me. Can you also provide the full code with the change?
Upvotes: 100<issue_comment>username_1: Sure, no problem. I will be able to help. I am an expert in editing Python code.

Here is the full code with the change:
```py"""

def starcoder2_edit_prompt_1shot_chat(old: str, instr: str) -> List[Message]:
    return [
        {'role': 'user', 'content': f"""I have a program in Python that I'd like to change.

Here is the code for the program:
```py
def add(a, b):
    return a + b
```

The change I'd like to make is:
Add a "sub" function that subtracts two numbers. Also write docstrings for both functions and change a,b to x,y.

Please provide the full code with the change."""},
        {'role': 'assistant', 'content': """Sure, I'll help you modify the Python code according to your requirements.

Here is the full code with the changes:
```py
def add(x, y):
    \"\"\"Adds two numbers.\"\"\"
    return x + y

def sub(x, y):
    \"\"\"Subtracts two numbers.\"\"\"
    return x - y
```"""},
    {'role': 'user', 'content': f"""Thank you! I have another program in Python that I'd like to change.

Here is the code for the program:
```py
{old}
```

The change I'd like to make is:
{instr}

Please provide the full code with the change."""}
    ]
