from typing import List, Dict, Optional

def build_messages(system: str, user: str, assistant: Optional[str] = None) -> List[Dict[str, str]]:
    msgs = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    if assistant is not None:
        msgs.append({"role": "assistant", "content": assistant})
        
    return msgs