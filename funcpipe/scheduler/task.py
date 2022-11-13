'''computation and communication tasks'''

class TaskManager:
    '''
    This class keeps track of the executed tasks and solves dependency issue
    '''
    def __init__(self):
        self.executed_task_ids = []

    def add_executed(self, task_id):
        self.executed_task_ids.append(task_id)

    def clear(self):
        self.executed_task_ids.clear()

    def check_dependency(self, ids) -> bool:
        check_result = True
        for id in ids:
            if id not in self.executed_task_ids:
                check_result = False
                break
        return check_result

# todo: wrap the task types as subclasses
class Task:
    # computation task
    FORWARD = 0
    BACKWARD = 1
    WEIGHT_UPDATE = 2
    # communication task
    RECV_INPUT = 3
    SEND_OUTPUT = 4
    RECV_GRAD = 5
    SEND_GRAD = 6
    SYNC_WEIGHTS = 7

    TYPES = [FORWARD, BACKWARD, WEIGHT_UPDATE,
             RECV_INPUT, SEND_OUTPUT, RECV_GRAD, SEND_GRAD, SYNC_WEIGHTS]
    COMP_TYPES = [FORWARD, BACKWARD, WEIGHT_UPDATE]
    COMM_TYPES = [RECV_INPUT, SEND_OUTPUT, RECV_GRAD, SEND_GRAD, SYNC_WEIGHTS]
    UPLOAD_TYPES = [SEND_OUTPUT, SEND_GRAD, SYNC_WEIGHTS]
    DOWNLOAD_TYPES = [RECV_INPUT, RECV_GRAD]
    SEND_OPS = [RECV_INPUT, RECV_GRAD]
    RECV_OPS = [SEND_OUTPUT, SEND_GRAD]

    def __init__(self, task_id, task_type):
        self.id = task_id
        self.dependencies = []
        self.comp_task_info = None
        self.comm_task_info = None  # a list of rank_ids that participates in the comm task
        self.micro_batch_id = -1

        if task_type not in Task.TYPES:
            raise Exception("Invalid task type specified!")

        self.type = task_type

    def __str__(self):
        type_str = ["FORWARD", "BACKWARD", "WEIGHT_UPDATE",
                    "RECV_INPUT", "SEND_OUTPUT", "RECV_GRAD", "SEND_GRAD", "SYNC_WEIGHTS"]

        def format_one(self):
            s = ""
            s += "--------\n"
            s += "Task type: %s\n" % type_str[self.type]
            if self.is_comp_task():
                t_info = "PartitionID:%d  MicrobatchID:%d\n" % (self.comp_task_info[0], self.comp_task_info[1])
            else:
                t_info = "Comm group ranks: %s\n" % str(self.comm_task_info)
            s +=  t_info
            s += "----------------------------\n"
            return s

        def format_two(self):
            s = ""
            s += "| Id: %d" % self.id
            s += "| Task type: %s" % type_str[self.type]
            if self.is_comp_task():
                t_info = "| PartitionID:%d  MicrobatchID:%d" % (self.comp_task_info[0], self.comp_task_info[1])
            else:
                t_info = "| Comm group ranks: %s" % str(self.comm_task_info)
            s += t_info
            s += "| dependencies: " + str(self.dependencies)
            s += "|"
            return s

        s = format_two(self)
        return s

    def set_micro_id(self, id):
        self.micro_batch_id = id

    def add_dependency(self, task_id):
        self.dependencies.append(task_id)

    def add_dependencies(self, task_ids):
        self.dependencies.extend(task_ids)

    def config_comp_task(self, partition_id, batch_id):
        if self.type not in Task.COMP_TYPES:
            raise Exception("Not a computation task!")
        self.comp_task_info = [partition_id, batch_id]

    def config_comm_task(self, rank_ids):
        self.comm_task_info = rank_ids
        #todo: further modification is required for the bubble free pipeline


    def is_comp_task(self) -> bool:
        return self.type in Task.COMP_TYPES

    def is_comm_task(self) -> bool:
        return self.type in Task.COMM_TYPES

    def is_upload_task(self) -> bool:
        return self.type in Task.UPLOAD_TYPES

    def is_download_task(self) -> bool:
        return self.type in Task.DOWNLOAD_TYPES