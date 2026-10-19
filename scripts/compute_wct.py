import sys
from shoreshop3_analysis import wct

if __name__ == '__main__':
    model = None
    if len(sys.argv) > 2:
        task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    print(task, ntasks)
    wct.MultiProfileWCT().compute(task, ntasks)