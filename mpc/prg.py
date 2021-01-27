import numpy as np


class PRG:
    def __init__(self: 'PRG', pid: int):
        self.pid = pid
        self.prg_states: dict = dict()

        np.random.seed()
        self.prg_states[self.pid] = np.random.get_state() 
        self.import_seed(-1, hash('global'))
        
        for other_pid in set(range(3)) - {self.pid}:
            self.import_seed(other_pid)
        
        self.switch_seed(self.pid)

    def import_seed(self: 'PRG', pid: int, seed: int = None):
        seed: int = hash((min(self.pid, pid), max(self.pid, pid))) if seed is None else seed
        seed %= (1 << 32)
        np.random.seed(seed)
        self.prg_states[pid] = np.random.get_state()
        
    def switch_seed(self: 'PRG', pid: int):
        self.prg_states[self.pid] = np.random.get_state()
        np.random.set_state(self.prg_states[pid])
    
    def restore_seed(self: 'PRG', pid: int):
        self.prg_states[pid] = np.random.get_state()
        np.random.set_state(self.prg_states[self.pid])
