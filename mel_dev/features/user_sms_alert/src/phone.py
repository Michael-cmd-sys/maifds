import re

def normalize_gh_number(raw: str) -> str:
    """
    Accepts:
      - 0559426442
      - +233559426442
      - 233559426442
      - 559426442
    Returns:
      - 233559426442  (digits only, E.164 without +)
    """
    if raw is None:
        raise ValueError("phone number is required")

    s = str(raw).strip()
    s = s.replace(" ", "").replace("-", "")
    s = s.replace("(", "").replace(")", "")

    # keep digits and optional leading +
    if s.startswith("+"):
        s = s[1:]

    # now digits only
    s = re.sub(r"\D+", "", s)

    if s.startswith("233") and len(s) >= 11:
        return s

    # local Ghana: 0XXXXXXXXX
    if s.startswith("0") and len(s) == 10:
        return "233" + s[1:]

    # sometimes user types 9 digits (no leading 0)
    if len(s) == 9:
        return "233" + s

    raise ValueError(f"Invalid Ghana MSISDN format: {raw}")
