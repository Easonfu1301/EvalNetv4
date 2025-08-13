




# print with red color
def rprint(*args, **kwargs):
    print("\033[91m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def gprint(*args, **kwargs):
    print("\033[92m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def yprint(*args, **kwargs):
    print("\033[93m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")


def bprint(*args, **kwargs):
    print("\033[94m", end="")
    print(*args, **kwargs)
    print("\033[0m", end="")