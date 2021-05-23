 
import os

blocks = [16, 32, 64, 128]
#trials = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
trials = [2048, 4096, 8192]

if __name__ == '__main__':
    cmd = "make montecarlo"
    os.system(cmd)
    
    for block in blocks:
        for trial in trials:
            cmd = f"./montecarlo {block} {trial}"
            os.system(cmd)

