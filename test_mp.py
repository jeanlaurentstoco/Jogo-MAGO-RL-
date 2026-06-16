import multiprocessing as mp
import os

if mp.current_process().name != 'MainProcess':
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

print(mp.current_process().name, os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set"))

def f(x):
    pass

if __name__ == '__main__':
    mp.set_start_method('spawn')
    p = mp.Process(target=f, args=(1,))
    p.start()
    p.join()
