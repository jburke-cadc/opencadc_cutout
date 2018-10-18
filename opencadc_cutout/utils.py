def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def is_integer(s):
    try:
        int(s)
        return True
    except ValueError:
        return False
