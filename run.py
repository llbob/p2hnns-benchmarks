from multiprocessing import freeze_support

from p2hnns_benchmarks.main import main

if __name__ == "__main__":
    freeze_support()
    main()
