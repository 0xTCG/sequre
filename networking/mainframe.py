import uuid
from typing import Callable, List, Any, Tuple
from functools import partial
from itertools import permutations

from networking.client import Client
from utils.constants import NUMBER_OF_CLIENTS, CP0, CP1, CP2
from mpc.secret import decompose, share_secret, load_shared_from_path, append_to_context, prune_context

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
    
    def __client_call(self: 'Mainframe', client: Client, func: Callable, context_id: int) -> Any:
        # Temp local scenario:
        return client.call(func, context_id)
    
    def __reconstruct_secret(self: 'Mainframe', values: List[tuple]) -> Any:
        public_value = values[0][0]
        shared_values = [val[1] for val in values]
        return public_value + sum(shared_values)

    def __set_shareds(self: 'Mainframe', *args, secret_args_mask: str,
                            context_id: Any, data_path: str) -> List[tuple]:
        if len(args) > 0:
            return [share_secret(
                        self.clients, arg,
                        context_id=context_id,
                        private=bool(int(mask)))
                    for arg, mask in zip(args, secret_args_mask)]
        return load_shared_from_path(self.clients, context_id, data_path)

    def __call(self: 'Mainframe', func: Callable) -> Callable:
        def secure_func(*args,
                        secret_args_mask: str = None,
                        preprocess: Callable = None,
                        data_path: str = None):
            context_id: int = uuid.uuid1()
            self.__set_shareds(
                *args,
                secret_args_mask=secret_args_mask,
                context_id=context_id,
                data_path=data_path)
            
            if preprocess is not None:
                self.__client_call(
                    client=self.preprocess_client,
                    func=preprocess,
                    context_id=context_id)
                
            returned_values: list = [
                self.__client_call(
                    client=client,
                    func=func,
                    context_id=context_id)
                for client in self.computing_clients]
            
            prune_context(clients=self.clients, context_id=context_id)
            return self.__reconstruct_secret(returned_values)
            
        return secure_func 
    
    __call__ = __call
