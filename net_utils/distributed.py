#!/usr/bin/env python3

import os
import torch
import signal
import random
import pickle
import threading
import traceback
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
mp.set_sharing_strategy('file_system')

"""Multiprocessing error handler."""

class ChildException(Exception):
    """Wraps an exception from a child process."""

    def __init__(self, child_trace):
        super(ChildException, self).__init__(child_trace)


class ErrorHandler(object):
    """Multiprocessing error handler (based on fairseq's).

    Listens for errors in child processes and
    propagates the tracebacks to the parent process.
    """

    def __init__(self, error_queue):
        # Shared error queue
        self.error_queue = error_queue
        # Children processes sharing the error queue
        self.children_pids = []
        # Start a thread listening to errors
        self.error_listener = threading.Thread(target=self.listen, daemon=True)
        self.error_listener.start()
        # Register the signal handler
        signal.signal(signal.SIGUSR1, self.signal_handler)

    def add_child(self, pid):
        """Registers a child process."""
        self.children_pids.append(pid)

    def listen(self):
        """Listens for errors in the error queue."""
        # Wait until there is an error in the queue
        child_trace = self.error_queue.get()
        # Put the error back for the signal handler
        self.error_queue.put(child_trace)
        # Invoke the signal handler
        os.kill(os.getpid(), signal.SIGUSR1)

    def signal_handler(self, sig_num, stack_frame):
        """Signal handler."""
        # Kill children processes
        for pid in self.children_pids:
            os.kill(pid, signal.SIGINT)
        # Propagate the error from the child process
        raise ChildException(self.error_queue.get())


"""Multiprocessing helpers."""

def run(proc_rank, world_size, port, error_queue, fun, fun_args, fun_kwargs):
    """Runs a function from a child process."""
    try:
        # Initialize the process group
        init_process_group(proc_rank, world_size, port)
        # Run the function
        fun(*fun_args, **fun_kwargs)
    except:
        # Propagate exception to the parent process
        error_queue.put(traceback.format_exc())
    finally:
        destroy_process_group()

def multi_proc_run(available_gpus, fun, fun_args=(), fun_kwargs={}):
    """Runs a function in a multi-proc setting."""

    # Handle errors from training subprocesses
    error_queue = mp.SimpleQueue()
    error_handler = ErrorHandler(error_queue)

    # Run each training subprocess
    port = random.randint(10001, 20002)
    ps = []
    num_proc = len(available_gpus)
    for i in available_gpus:
        p_i = mp.Process(
            target=run,
            args=(i, num_proc, port, error_queue, fun, fun_args, fun_kwargs)
        )
        ps.append(p_i)
        p_i.start()
        error_handler.add_child(p_i.pid)

    # Wait for each subprocess to finish
    for p in ps:
        p.join()

"""Distributed helpers."""

def is_master_proc(num_gpus, config):
    """Determines if the current process is the master process.

    Master process is responsible for logging, writing and loading checkpoints.
    In the multi GPU setting, we assign the master role to the rank 0 process.
    When training using a single GPU, there is only one training processes
    which is considered the master processes.

    Modified: when using multiple GPUs, the first in available list is the main 
    process.
    """
    return num_gpus == 1 or torch.distributed.get_rank() == config.distributed.available_gpus[0]

def get_world_size():
    if not dist.is_available():
        return 1
    if not dist.is_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def all_gather_differentiable(tensor):
    """
        Run differentiable gather function for SparseConv features with variable number of points.
        tensor: [num_points, feature_dim]
    """
    world_size = get_world_size()
    if world_size == 1:
        return [tensor]

    num_points, f_dim = tensor.size()
    local_np = torch.LongTensor([num_points]).to("cuda")
    np_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(np_list, local_np)
    np_list = [int(np.item()) for np in np_list]
    max_np = max(np_list)

    tensor_list = []
    for _ in np_list:
        tensor_list.append(torch.FloatTensor(size=(max_np, f_dim)).to("cuda"))
    if local_np != max_np:
        padding = torch.zeros(size=(max_np-local_np, f_dim)).to("cuda").float()
        tensor = torch.cat((tensor, padding), dim=0)
        assert tensor.size() == (max_np, f_dim)

    dist.all_gather(tensor_list, tensor)

    data_list = []
    for gather_np, gather_tensor in zip(np_list, tensor_list):
        gather_tensor = gather_tensor[:gather_np]
        assert gather_tensor.size() == (gather_np, f_dim)
        data_list.append(gather_tensor)
    return data_list


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # obtain Tensor size of each rank
    local_size = torch.LongTensor([tensor.numel()]).to("cuda")
    size_list = [torch.LongTensor([0]).to("cuda") for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.ByteTensor(size=(max_size,)).to("cuda"))
    if local_size != max_size:
        padding = torch.ByteTensor(size=(max_size - local_size,)).to("cuda")
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def init_process_group(proc_rank, world_size, port):
    """Initializes the default process group."""
    # Set the GPU to use
    print(proc_rank)

    torch.cuda.set_device(proc_rank)
    # Initialize the process group
    torch.distributed.init_process_group(
        backend="nccl",
        init_method="tcp://{}:{}".format("localhost", port),
        world_size=world_size,
        rank=proc_rank
    )

def destroy_process_group():
    """Destroys the default process group."""
    torch.distributed.destroy_process_group()

def initiate_environment(config):
    seed = config.seed + get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def reduce_dict(input_dict, average=True):
    """
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict