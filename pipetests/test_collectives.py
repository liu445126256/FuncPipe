import numpy as np
import threading

from funcpipe.comm.collectives import P2P, One2Many, AllReduce
from funcpipe.platforms import Platform
from funcpipe.debugger import Logger

def send_and_recv():
    send_rank = 0
    recv_rank = 1
    data = np.array([1, 2, 3, 4])
    P2P.send(send_rank, recv_rank, data, comm_mark="test")
    recv_data = P2P.recv(recv_rank, send_rank, comm_mark="test")
    Logger.debug(str(recv_data))

def one_to_many_send():
    pass

def one_to_many_recv():
    pass

def allreduce(rank_num):
    thds = []
    for tid in range(rank_num):
        thd = threading.Thread(target=_allreduce_thread, args=(tid, list(range(rank_num)), ))
        thd.start()
        thds.append(thd)
    for thd in thds: thd.join()

def _allreduce_thread(rank_id, comm_ranks):
    data = np.array([rank_id for i in range(10)])
    reduce_data = AllReduce.latency_opt(rank_id, comm_ranks, data)
    Logger.info(("rank%d: " % rank_id)+ str(reduce_data))


if __name__ == "__main__":
    Platform.use("ali")
    Logger.use_logger(Logger.NATIVE, Logger.INFO, "collective_test")

    #send_and_recv()
    allreduce(4)
