from typing import List, Sequence, Tuple, Dict, Union
import numpy as np
import pickle
import threading
import time
from collections import deque
import gc

from funcpipe.platforms import Platform
from funcpipe.debugger import Logger
from funcpipe.monitor import Monitor
from funcpipe.timeline import Timeline


class P2P:
    '''P2P data transfer'''

    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def send(my_rank, dst_rank, data: Union[np.ndarray, List[np.ndarray]], comm_mark: str) -> None:
        file_name = '{}_{}_{}.npy'.format(my_rank, dst_rank, comm_mark)
        data_in_bytes = pickle.dumps(data)
        Platform.upload_to_storage(file_name, data_in_bytes)
        Logger.debug("P2P send: %s" % file_name)

    @staticmethod
    def recv(my_rank, src_rank, comm_mark: str) -> Union[np.ndarray, List[np.ndarray]]:
        file_name = '{}_{}_{}.npy'.format(src_rank, my_rank, comm_mark)
        Logger.debug("P2P recv: %s" % file_name)
        try:
            data = pickle.loads(Platform.download_from_storage(file_name))
            Platform.delete_from_storage(file_name)
            return data
        except:
            raise Exception("No corresponding file found Or platform raise an error!")


class One2Many:
    '''one to many communication'''

    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def send(my_rank: int, dst_ranks: List, data: Union[np.ndarray, List[np.ndarray]], comm_mark: str) -> None:
        # send in parallel to hide latency
        comm_threads = []
        for dst_rank in dst_ranks:
            comm_thd = threading.Thread(target=P2P.send, args=(my_rank, dst_rank, data, comm_mark,))
            comm_threads.append(comm_thd)
        for comm_thd in comm_threads: comm_thd.join()

    @staticmethod
    def recv(my_rank: int, src_ranks: List, comm_mark: str) -> Union[np.ndarray, List[np.ndarray]]:
        # recv in parallel to hide latency
        comm_threads = []
        recv_parts = {}
        for src_rank in src_ranks:
            comm_thd = threading.Thread(target=One2Many.recv_thd_func, args=(my_rank, src_rank, recv_parts, comm_mark,))
            comm_threads.append(comm_thd)
        for comm_thd in comm_threads: comm_thd.join()
        data = np.zeros((1))
        for k in recv_parts.keys():
            data = np.concatenate(data, recv_parts[k])
        data = np.delete(data, 0)
        return data

    @staticmethod
    def recv_thd_func(my_rank: int, src_rank: int, recv_data: Dict, comm_mark: str) -> None:
        data = P2P.recv(my_rank, src_rank, comm_mark)
        recv_data[src_rank] = data


class AllReduce:
    '''all reduce communication'''

    def __init__(self) -> None:
        raise Exception("This is a static class and should not have any instance.")

    @staticmethod
    def latency_opt(my_rank, comm_ranks, data: np.ndarray, comm_marker: str = None) -> np.ndarray:
        Logger.debug("Allreduce start.")
        Monitor.print_my_processs_mem("allreduce start mem")
        assert len(data.shape) == 1  # 1-d array required
        num_elements = data.shape[0]
        num_ranks = len(comm_ranks)
        elements_per_worker = num_elements // num_ranks
        tail_n = num_elements % num_ranks

        sorted_ranks = comm_ranks.copy()
        sorted_ranks.sort()

        # need to generate an unique marker
        # todo: consider making all collectives comm_marker requirement consistent
        if not comm_marker:
            comm_marker = ''
            for rank in comm_ranks: comm_marker += str(rank)

        my_comm_id = sorted_ranks.index(my_rank)
        my_offset = (elements_per_worker * my_comm_id) + min(tail_n, my_comm_id)
        my_length = elements_per_worker + (1 if my_comm_id < tail_n else 0)
        # my_chunk = data[my_offset: my_offset + my_length]
        result = np.zeros(num_elements, dtype=data.dtype)

        Timeline.start("a1(upload local chuncks)")
        gc.collect()
        Monitor.print_my_processs_mem("a1")
        upload_threads = []
        for i in range(num_ranks):
            if i != my_comm_id:
                offset = (elements_per_worker * i) + min(tail_n, i)
                length = elements_per_worker + (1 if i < tail_n else 0)
                file_name = "{}_{}_{}_{}".format("allreduce", i, my_comm_id,
                                                 comm_marker)  # distinct comm_marker required to seperate different allreduce ops
                data_in_bytes = pickle.dumps(data[offset: offset + length])
                # Monitor.print_my_processs_mem("pickle dumps:{}".format(len(data_in_bytes)))
                Platform.upload_to_storage(file_name, data_in_bytes)
                # Monitor.print_my_processs_mem("rank{}".format(i))
                '''
                thd = threading.Thread(target=Platform.upload_to_storage, args=(file_name, data_in_bytes, ))
                thd.start()
                upload_threads.append(thd)
                '''
                # del data_in_bytes
                # gc.collect()
        for thd in upload_threads: thd.join()  # todo: parallel upload to hide latency
        Timeline.end("a1(upload local chuncks)")

        # read all chunks: id = my_id from storage
        Timeline.start("a2(download)")
        gc.collect()
        Monitor.print_my_processs_mem("-a2")

        '''
        download_threads = []
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}_{}".format("allreduce", my_comm_id, comm_id, comm_marker)
            Timeline.start("download and load", silent=True)
            recv_data = pickle.loads(Platform.download_from_storage(file_name))
            Timeline.end("download and load", silent=True)
            #Monitor.print_my_processs_mem("pickle load:{}".format(len(recv_data)))
            Timeline.start("delete", silent=True)
            Platform.delete_from_storage(file_name)
            Timeline.end("delete", silent=True)
            #todo: stucks here when using ali cloud (np.add stucks, not sure why)
            Timeline.start("a2-1(merge)", silent=True)
            #Monitor.print_my_processs_mem("before a2-1 merge")

            result[my_offset:my_offset + my_length] = result[my_offset:my_offset + my_length] + recv_data
            #my_chunk = my_chunk + recv_data # a data copy is generated when the data is modified?

            #Monitor.print_my_processs_mem("after a2-1 merge")
            Timeline.end("a2-1(merge)", silent=True)

            Timeline.start("collect", silent=True)
            del recv_data
            gc.collect()
            Timeline.end("collect", silent=True)
        for thd in download_threads: thd.join() #todo: parallel download to hide latency
        '''

        files_to_download = []
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}_{}".format("allreduce", my_comm_id, comm_id, comm_marker)
            files_to_download.append(file_name)
        file_status = [0 for f in files_to_download]
        while sum(file_status) < len(file_status):
            for i, file_name in enumerate(files_to_download):
                if file_status[i]: continue
                if Platform.file_exists(file_name):
                    Timeline.start("download and load", silent=True)
                    recv_data = pickle.loads(Platform.download_from_storage(file_name))
                    Timeline.end("download and load", silent=True)
                    Timeline.start("delete", silent=True)
                    Platform.delete_from_storage(file_name)
                    Timeline.end("delete", silent=True)
                    # todo: stucks here when using ali cloud (np.add stucks, not sure why)
                    Timeline.start("a2-1(merge)", silent=True)
                    result[my_offset:my_offset + my_length] = result[my_offset:my_offset + my_length] + recv_data
                    Timeline.end("a2-1(merge)", silent=True)
                    del recv_data
                    gc.collect()
                    file_status[i] = 1

        Timeline.end("a2(download)")

        # write the merged chunk to storage
        Timeline.start("a3(upload merged)")
        gc.collect()
        Monitor.print_my_processs_mem("a3")
        # todo: ensure the filename does not collide with other ops
        file_name = "{}_{}_{}".format("allreduce", my_comm_id, comm_marker)
        data_in_bytes = pickle.dumps(result[my_offset:my_offset + my_length])
        Monitor.print_my_processs_mem("pickle dumps:{}".format(len(data_in_bytes)))
        Platform.upload_to_storage(file_name, data_in_bytes)
        del data_in_bytes
        gc.collect()
        Timeline.end("a3(upload merged)")

        # read other merged chunks
        Timeline.start("a4(download other merged)")

        '''
        gc.collect()
        Monitor.print_my_processs_mem("a4")
        #merged_chunks = dict()
        #merged_chunks[my_comm_id] = my_chunk
        download_threads = []
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
            recv_data = pickle.loads(Platform.download_from_storage(file_name)) # debug: to remove
            #Logger.debug(str(len(recv_bytes))) # debug: to remove
            #Monitor.print_my_processs_mem("Downloaded:{}".format(len(recv_bytes)))
            #recv_data = pickle.loads(recv_bytes)
            #Monitor.print_my_processs_mem("pickle load")

            #merged_chunks[comm_id] = recv_data
            peer_offset = (elements_per_worker * comm_id) + min(tail_n, comm_id)
            peer_length = elements_per_worker + (1 if comm_id < tail_n else 0)
            result[peer_offset:peer_offset + peer_length] = recv_data

            #del recv_bytes
            del recv_data
            gc.collect()
        for thd in download_threads: thd.join() # todo: parallel download to hide latency
        '''
        files_to_download = []
        file_status = []  # use comm_id for status, -1 for finished
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
            files_to_download.append(file_name)
            file_status.append(comm_id)
        while sum(file_status) > len(file_status) * (-1):
            for i, file_name in enumerate(files_to_download):
                if file_status[i] == -1: continue
                if Platform.file_exists(file_name):
                    comm_id = file_status[i]
                    recv_data = pickle.loads(Platform.download_from_storage(file_name))
                    # merged_chunks[comm_id] = recv_data
                    peer_offset = (elements_per_worker * comm_id) + min(tail_n, comm_id)
                    peer_length = elements_per_worker + (1 if comm_id < tail_n else 0)
                    result[peer_offset:peer_offset + peer_length] = recv_data
                    del recv_data
                    gc.collect()
                    file_status[i] = -1

        Timeline.end("a4(download other merged)")

        # reconstruct
        # todo: avoid using np.concatenate
        '''
        result = merged_chunks[0]
        for k in range(1, num_ranks):
            result = np.concatenate((result, merged_chunks[k]))
        '''

        '''
        Timeline.start("a5(reconstruct)")
        gc.collect()
        Monitor.print_my_processs_mem("a5")
        result = np.zeros(num_elements)
        offset = 0
        for k in range(1, num_ranks):
            result[offset: offset + len(merged_chunks[k])] = merged_chunks[k]
            offset += len(merged_chunks[k])
        Timeline.end("a5(reconstruct)")
        '''

        # rank 0 deletes the intermediate files
        if my_comm_id == 0:
            for comm_id in range(1, num_ranks):
                flag_name = "{}_{}_{}".format("allreduceDone", comm_id, comm_marker)
                while not Platform.file_exists(flag_name):
                    time.sleep(0.06)
            for comm_id in range(num_ranks):
                flag_name = "{}_{}_{}".format("allreduceDone", comm_id, comm_marker)
                file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
                Platform.delete_from_storage(file_name)
                if comm_id > 0: Platform.delete_from_storage(flag_name)
        else:
            flag_name = "{}_{}_{}".format("allreduceDone", my_comm_id, comm_marker)
            Platform.upload_to_storage(flag_name, bytes(1))
        Logger.debug("Allreduce end.")
        Monitor.print_my_processs_mem("end")
        return result

    @staticmethod
    def bandwidth_opt(my_rank, comm_ranks, data: np.ndarray, comm_marker: str = None) -> np.ndarray:
        Logger.debug("Allreduce start.")
        assert len(data.shape) == 1  # 1-d array required
        num_elements = data.shape[0]
        num_ranks = len(comm_ranks)
        elements_per_worker = num_elements // num_ranks
        tail_n = num_elements % num_ranks

        sorted_ranks = comm_ranks.copy()
        sorted_ranks.sort()

        # need to generate an unique marker
        # todo: consider making all collectives comm_marker requirement consistent
        if not comm_marker:
            comm_marker = ''
            for rank in comm_ranks: comm_marker += str(rank)

        my_comm_id = sorted_ranks.index(my_rank)
        my_offset = (elements_per_worker * my_comm_id) + min(tail_n, my_comm_id)
        my_length = elements_per_worker + (1 if my_comm_id < tail_n else 0)
        # my_chunk = data[my_offset: my_offset + my_length]
        result = np.zeros(num_elements, dtype=data.dtype)

        ###########################################################
        # make use of download / upload bandwidth simultaneously
        def upload_thd(task_queue: deque):
            while len(task_queue) > 0:
                info = task_queue.popleft()
                # data = info[0]
                file_name = info[1]
                offset = info[2]
                offset_end = info[3]
                data_in_bytes = pickle.dumps(data[offset: offset_end])
                Platform.upload_to_storage(file_name, data_in_bytes)
                del data_in_bytes
                gc.collect()

        def download_thd(task_queue: deque):
            # version 1
            '''
            while len(task_queue) > 0:
                info = task_queue.popleft()
                #my_chunk = info[0]
                file_name = info[1]
                recv_data = pickle.loads(Platform.download_from_storage(file_name))
                Platform.delete_from_storage(file_name)
                # todo: stucks here when using ali cloud (np.add stucks, not sure why)
                #my_chunk[:] = my_chunk + recv_data
                result[my_offset:my_offset + my_length] = result[my_offset:my_offset + my_length] + recv_data
                del recv_data
                gc.collect()
            '''

            # version 2
            files_to_download = []
            while len(task_queue) > 0:
                info = task_queue.popleft()
                file_name = info[1]
                files_to_download.append(file_name)
            file_status = [0 for f in files_to_download]
            while sum(file_status) < len(file_status):
                for i, file_name in enumerate(files_to_download):
                    if file_status[i]: continue
                    if Platform.file_exists(file_name):
                        recv_data = pickle.loads(Platform.download_from_storage(file_name))
                        Platform.delete_from_storage(file_name)
                        # todo: stucks here when using ali cloud (np.add stucks, not sure why)
                        result[my_offset:my_offset + my_length] = result[my_offset:my_offset + my_length] + recv_data
                        del recv_data
                        gc.collect()
                        file_status[i] = 1

        Timeline.start("a1+2(upload + download)")
        Monitor.print_my_processs_mem("a1+2")
        upload_queue = deque()
        for i in range(num_ranks - 1):
            chunk_id = (my_comm_id + (1 + i)) % num_ranks
            offset = (elements_per_worker * chunk_id) + min(tail_n, chunk_id)
            length = elements_per_worker + (1 if chunk_id < tail_n else 0)
            file_name = "{}_{}_{}_{}".format("allreduce", chunk_id, my_comm_id,
                                             comm_marker)  # distinct comm_marker required to seperate different allreduce ops
            upload_queue.append((None, file_name, offset, offset + length))

        # read all chunks: id = my_id from storage
        download_queue = deque()
        for i in range(num_ranks - 1):
            chunk_id = (my_comm_id - 1 - i + num_ranks) % num_ranks
            file_name = "{}_{}_{}_{}".format("allreduce", my_comm_id, chunk_id, comm_marker)
            download_queue.append((None, file_name))

        upload_thread = threading.Thread(target=upload_thd, args=(upload_queue,))
        download_thread = threading.Thread(target=download_thd, args=(download_queue,))
        upload_thread.start()
        download_thread.start()
        upload_thread.join()
        download_thread.join()
        #########################################################
        Timeline.end("a1+2(upload + download)")

        # write the merged chunk to storage
        # todo: ensure the filename does not collide with other ops
        Timeline.start("a3(upload merged)")
        Monitor.print_my_processs_mem("a3")
        file_name = "{}_{}_{}".format("allreduce", my_comm_id, comm_marker)
        data_in_bytes = pickle.dumps(result[my_offset:my_offset + my_length])
        Platform.upload_to_storage(file_name, data_in_bytes)
        del data_in_bytes
        gc.collect()
        Timeline.end("a3(upload merged)")

        # read other merged chunks
        # merged_chunks = dict()
        # merged_chunks[my_comm_id] = my_chunk
        Timeline.start("a4(download other merged)")
        Monitor.print_my_processs_mem("a4")

        '''
        download_threads = []
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
            # recv_bytes = Platform.download_from_storage(file_name) # debug: to remove
            recv_data = pickle.loads(Platform.download_from_storage(file_name))
            # merged_chunks[comm_id] = recv_data
            peer_offset = (elements_per_worker * comm_id) + min(tail_n, comm_id)
            peer_length = elements_per_worker + (1 if comm_id < tail_n else 0)
            result[peer_offset:peer_offset + peer_length] = recv_data
            del recv_data
            gc.collect()
        for thd in download_threads: thd.join()  # todo: parallel download to hide latency
        '''

        files_to_download = []
        file_status = []  # use comm_id for status, -1 for finished
        for comm_id in range(num_ranks):
            if comm_id == my_comm_id: continue
            file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
            files_to_download.append(file_name)
            file_status.append(comm_id)
        while sum(file_status) > len(file_status) * (-1):
            for i, file_name in enumerate(files_to_download):
                if file_status[i] == -1: continue
                if Platform.file_exists(file_name):
                    comm_id = file_status[i]
                    recv_data = pickle.loads(Platform.download_from_storage(file_name))
                    # merged_chunks[comm_id] = recv_data
                    peer_offset = (elements_per_worker * comm_id) + min(tail_n, comm_id)
                    peer_length = elements_per_worker + (1 if comm_id < tail_n else 0)
                    result[peer_offset:peer_offset + peer_length] = recv_data
                    del recv_data
                    gc.collect()
                    file_status[i] = -1
        Timeline.end("a4(download other merged)")

        # rank 0 deletes the intermediate files
        if my_comm_id == 0:
            for comm_id in range(1, num_ranks):
                flag_name = "{}_{}_{}".format("allreduceDone", comm_id, comm_marker)
                while not Platform.file_exists(flag_name):
                    time.sleep(0.06)
            for comm_id in range(num_ranks):
                flag_name = "{}_{}_{}".format("allreduceDone", comm_id, comm_marker)
                file_name = "{}_{}_{}".format("allreduce", comm_id, comm_marker)
                Platform.delete_from_storage(file_name)
                if comm_id > 0: Platform.delete_from_storage(flag_name)
        else:
            flag_name = "{}_{}_{}".format("allreduceDone", my_comm_id, comm_marker)
            Platform.upload_to_storage(flag_name, bytes(1))
        Logger.debug("Allreduce end.")
        Monitor.print_my_processs_mem("end")

        return result
