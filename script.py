 
import os

threads = [1, 2, 4, 6, 8]
trials = [1, 10, 100, 1000, 10000, 100000, 500000]


for th in threads:
    for tr in trials:
        #print(f"NUMT = {t}, NUMTRIALS = {s}")
        cmd = f"g++ -DNUMT={th} -DNUMTRIALS={tr} project5.cpp -o prog -lm -fopenmp"
        os.system(cmd)
        cmd = "./prog"
        os.system(cmd)

