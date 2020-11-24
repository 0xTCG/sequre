from typing import Callable, Any
from collections import defaultdict

from utils.constants import CP0, CP1, CP2

class Client:
    # TODO: Make this a connection class only. Separate server specific stuff into a different class.
    def __init__(self: 'Client', pid: int):
        self.pid = pid
        self.__context = defaultdict(list)
        self.__other_clients = dict()
    
    def client_connect(self: 'Client', other: 'Client'):
        # Temp local solution
        self.__other_clients[other.pid] = other
    
    def client_disconnect(self: 'Client', other: 'Client'):
        # Temp local solution:
        del self.__other_clients[other.pid]

    def call(self: 'Client', secure_fn: Callable, *args) -> Any:
        # TODO: Implement async/sync callstack
        
        return secure_fn(*args)
    
    def append_to_context(self: 'Client', context_id: int, shared: Any):
        self.__context[context_id].append(shared)
    
    def prune_context(self: 'Client', context_id: int):
        del self.__context[context_id]
    
    def get_shared(self: 'Client', context_id: int, secrets: list, transform: Callable) -> Any:
        return transform([self.__context[context_id][i] for i in secrets])

    def get_counter_client(self: 'Client') -> 'Client':
        if self.pid == CP0:
            raise ValueError('Preprocess client has no counter client.')
        
        return self.__other_clients[int(not self.pid)]

    def reconstruct_secret(self: 'Client', context_id: int, secrets: list, transform: Callable) -> Any:
        self_shared = self.get_shared(context_id, secrets, transform)
        other_shared = self.get_counter_client().get_shared(context_id, secrets, transform)

        return self_shared + other_shared
    
    def load_shared_from_path(self: 'Client', context_id: Any, data_path: str) -> list:
        raise NotImplementedError()
