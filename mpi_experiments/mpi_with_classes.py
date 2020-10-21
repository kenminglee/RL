from mpi4py import MPI
import torch.multiprocessing as mp
from base_agent import base_agent

''' Goal: Create 4 worker processes, each of them count to 1000. 
For every 100 numbers they count, send counter to main process. 
Wait until all 4 processes have sent results to main process, then
Main process will broadcast a number that the worker process will need to continue counting from
Terminate once everyone reached 1000'''
comm = MPI.COMM_WORLD
# get number of processes
num_proc = comm.Get_size()
# get pid
rank = comm.Get_rank()

class Counter(base_agent):
    def __init__(self):
        self.counter = 0
    
    def choose_action(self, s) -> int:
        pass

    def learn(self, s, a, r, s_, done) -> int:
        pass
    
    def count_hundred(self):
        for i in range(100):
            self.counter += 1
        return self.counter

def count_hundred(counter):
    for i in range(100):
        counter += 1
    return counter

if __name__=='__main__':
    
    print(rank)
    score = 0
    obj = Counter()
    while score < 1000:
        counter = obj.count_hundred()
        # Root process waits until counter is obtained from all processes (blocking!)
        data = comm.gather(counter, root=0)
        if rank == 0:
            assert len(data)==num_proc
            score = data[0]
        # Blocking call - waits until root is done processing data and broadcast results to everyone else
        score = comm.bcast(score, root=0)
        print(score)
