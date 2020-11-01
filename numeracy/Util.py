# Created by lyc at 2020/10/16 15:31


def require(expr: bool, message = "Requirement is not satisfied!"):
    if not expr:
        raise Exception(message)

