from datetime import datetime

def result_name(result, when, number):
    return f"{result}_{when:%Y%m%d_%H%M%S}_{number:03d}.png"

t = datetime(2026, 11, 22, 14, 5, 9)
print(result_name("NG", t, 7))
print(result_name("OK", t, 12))


# NG_20261122_140509_007.png
# OK_20261122_140509_012.png