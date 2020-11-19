from typing import Callable, List, Any

from networking.client import Client
from utils.constants import NUMBER_OF_CLIENTS
from mpc.secret import share_secret, reconstruct_secret

class Mainframe:
    def __init__(self: 'Mainframe'):
        self.clients = [self.__connect_to_client(i) for i in range(NUMBER_OF_CLIENTS)]
    
    def __enter__(self: 'Mainframe'):
        # TODO: Establish connections between clients
        # TODO: Secret share the data between clients (if any)
        return self

    def __exit__(self: 'Mainframe', exc_type, exc_val, exc_tb):
        # TODO: Close all connections
        pass

    def __repr__(self: 'Mainframe') -> Callable:
        return self.__call
    
    def __connect_to_client(self: 'Mainframe', pid: int) -> Client:
        # TODO: Establish connection.
        
        # Temp local scenario:
        return Client(pid)
    
    def __client_call(self: 'Mainframe', client: Client, func: Callable, *args) -> Any:
        # Temp local scenario
        return client.call(func, *args)
    
    def __call(self: 'Mainframe', func: Callable) -> Callable:
        def secure_func(*args, args_mask: str): 
            shared_args: List[tuple] = [
                share_secret(arg) if int(mask) else (arg, arg)
                for arg, mask in zip(args, args_mask)]
                
            returned_values: list = [
                self.__client_call(client, func, *[arg[i] for arg in shared_args])
                for i, client in enumerate(self.clients[1:])]
            
            return reconstruct_secret(returned_values)
            
        return secure_func 
    
    __call__ = __call
