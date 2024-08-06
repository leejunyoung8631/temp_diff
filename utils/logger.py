"""
CELL Logger
Yeseong Kim @ CELL DGIST, 2022
"""


class Logger:
    """ CELL Logger """
    def __init__(self, filename, verbose=True):
        self.f = open(filename, 'w')  # pylint: disable=consider-using-with
        self.verbose = verbose

    def __del__(self):
        self.f.close()

    def i(self, *args):
        for i, arg in enumerate(args):
            if i > 0:
                self.f.write(', ')
            self.f.write(str(arg) + '')
        self.f.write('\n')

    def d(self, *args):
        self.i(*args)
        if self.verbose:
            print(*args)


# Global log instance
def sLog():
    assert sLog.instance is not None, "Call init_sLog() first"
    return sLog.instance


sLog.instance = None


def init_sLog(filename, verbose=True):
    sLog.instance = Logger(filename, verbose)


# Try to print in sLog; if unavailable, use print
def print_log(*args):
    if sLog.instance is None:
        print(*args)
    else:
        sLog.instance.d(*args)
