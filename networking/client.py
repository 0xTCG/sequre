from typing import Callable, Any


class Client:
    def __init__(self: 'Client', pid: int):
        # TODO: Implement Client class as server

        self.pid = pid


    def call(self: 'Client', secure_fn: Callable, *args) -> Any:
        # TODO: Implement async/sync callstack
        
        return secure_fn(*args)
