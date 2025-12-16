from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

@dataclass
class SmsSendResult:
    status_code: int
    success: bool
    provider_code: Optional[str]
    provider_message: Optional[str]
    raw_body: Optional[str]
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
