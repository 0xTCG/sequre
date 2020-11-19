import uuid
from typing import Callable, List, Any
from functools import partial
from itertools import permutations

from networking.client import Client
from utils.constants import NUMBER_OF_CLIENTS, CP0, CP1, CP2
from mpc.secret import share_secret

class Mainframe:
    def __init__(self: 'Mainframe'):
        self.clients = [self.__connect_to_client(i) for i in range(NUMBER_OF_CLIENTS)]
        self.computing_clients = [self.clients[i] for i in [CP1, CP2]]
        self.preprocess_client = self.clients[CP0]
    
    def __enter__(self: 'Mainframe'):
        for c_1, c_2 in permutations(self.clients, 2):
            c_1.client_connect(c_2)
        
        # TODO: Secret share the data between clients (if any)
        return self

    def __exit__(self: 'Mainframe', exc_type, exc_val, exc_tb):
        for c_1, c_2 in permutations(self.clients, 2):
            c_1.client_disconnect(c_2)

    def __repr__(self: 'Mainframe') -> Callable:
        return self.__call
    
    def __connect_to_client(self: 'Mainframe', pid: int) -> Client:
        # TODO: Establish connection.
        
        # Temp local scenario:
        return Client(pid)
    
    def __client_call(self: 'Mainframe', client: Client, func: Callable, *args) -> Any:
        # Temp local scenario:
        return client.call(func, *args)
    
    def __append_to_context(self: 'Mainframe', context_id: int, shared_pair: tuple):
        shared_1, shared_2 = shared_pair
        self.clients[CP1].append_to_context(context_id, shared_1)
        self.clients[CP2].append_to_context(context_id, shared_2)

    def __share_secret(self: 'Mainframe', value: Any, context_id: int, mask: int) -> tuple:
        shared_pair: tuple = share_secret(value) if mask else (value, ) * 2
        self.__append_to_context(context_id, shared_pair)

        return shared_pair
    
    def __prune_context(self: 'Mainframe', context_id: int):
        for client in self.computing_clients:
            client.prune_context(context_id)
    
    def __reconstruct_secret(self: 'Mainframe', values: List[tuple]) -> Any:
        public_value = values[0][0]
        shared_values = [val[1] for val in values]
        return public_value + sum(shared_values)

    def __call(self: 'Mainframe', func: Callable) -> Callable:
        def secure_func(*args, secret_args_mask: str, preprocess: Callable): 
            context_id: int = uuid.uuid1()
            shared_args: List[tuple] = [
                self.__share_secret(
                    arg,
                    context_id=context_id,
                    mask=int(mask))
                for arg, mask in zip(args, secret_args_mask)]
            
            if preprocess:
                extra_args: List[tuple] = self.__client_call(
                    client=self.preprocess_client,
                    func=preprocess)
                for shared_pair in extra_args:
                    self.__append_to_context(context_id, shared_pair)
                shared_args.extend(extra_args)
                
            returned_values: list = [
                self.__client_call(
                    client,
                    partial(func, client=client, context_id=context_id),
                    *[arg[i] for arg in shared_args])
                for i, client in enumerate(self.computing_clients)]
            
            self.__prune_context(context_id)
            return self.__reconstruct_secret(returned_values)
            
        return secure_func 
    
    __call__ = __call
