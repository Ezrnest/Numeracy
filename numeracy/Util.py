# Created by lyc at 2020/10/16 15:31


def require(expr: bool):
    if not expr:
        raise Exception("Requirement is not satisfied!")
