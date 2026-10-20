import sys
from shoreshop3_analysis import wct

if __name__ == '__main__':
    model = None
    task = 0
    ntasks = 1
    if len(sys.argv) > 2:
        task, ntasks = int(sys.argv[1]), int(sys.argv[2])
    wct.ModelsWCT().compute(task, ntasks)