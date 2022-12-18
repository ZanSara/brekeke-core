import sys
from listener import Listener

if __name__ == "__main__":
    Listener(model=sys.argv[1]).listen()
