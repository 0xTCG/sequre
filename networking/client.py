from typing import Callable, Any
from collections import defaultdict

from utils.constants import CP0, CP1, CP2

class Client:
    # TODO: Make this a connection class only. Separate server specific stuff into a different class.
    def __init__(self: 'Client', pid: int):
        self.pid = pid
        self.__context = defaultdict(list)
        self.__other_clients = dict()

        self._construct_cache()
    
    def _construct_lagrange_cache(self: 'Client'):
        """
        Mat<ZZ_p> table;

        // Table 0
        table.SetDims(1, 2);
        if (pid > 0) {
            table[0][0] = 1;
            table[0][1] = 0;
        }
        table_type_ZZ[0] = true;
        table_cache[0] = table;
        table_field_index[0] = 2;

        // Table 1
        int half_len = Param::NBIT_K / 2;
        table.SetDims(2, half_len + 1);
        if (pid > 0) {
            for (int i = 0; i < half_len + 1; i++) {
            if (i == 0) {
                table[0][i] = 1;
                table[1][i] = 1;
            } else {
                table[0][i] = table[0][i - 1] * 2;
                table[1][i] = table[1][i - 1] * 4;
            }
            }
        }
        table_type_ZZ[1] = true;
        table_cache[1] = table;
        table_field_index[1] = 1;

        // Table 2: parameters (intercept, slope) for piecewise-linear approximation
        // of
        //          negative log-sigmoid function
        table.SetDims(2, 64);
        if (pid > 0) {
            ifstream ifs;
            ifs.open("sigmoid_approx.txt");
            if (!ifs.is_open()) {
            DBG("Error opening sigmoid_approx.txt");
            clear(table);
            }
            for (int i = 0; i < table.NumCols(); i++) {
            double intercept, slope;
            ifs >> intercept >> slope;

            ZZ_p fp_intercept, fp_slope;
            DoubleToFP(fp_intercept, intercept, Param::NBIT_K, Param::NBIT_F);
            DoubleToFP(fp_slope, slope, Param::NBIT_K, Param::NBIT_F);

            table[0][i] = fp_intercept;
            table[1][i] = fp_slope;
            }
            ifs.close();
        }
        table_type_ZZ[2] = false;
        table_cache[2] = table;
        table_field_index[2] = 0;

        // DBG("Generating lagrange cache");

        for (int cid = 0; cid < table_cache.length(); cid++) {
            long nrow = table_cache[cid].NumRows();
            long ncol = table_cache[cid].NumCols();
            bool index_by_ZZ = table_type_ZZ[cid];
            if (index_by_ZZ) {
            lagrange_cache[cid].SetDims(nrow, 2 * ncol);
            } else {
            lagrange_cache[cid].SetDims(nrow, ncol);
            }

            if (pid > 0) {
            // DBG("Lagrange interpolation for Table {}... ",cid);
            for (int i = 0; i < nrow; i++) {
                Vec<long> x;
                Vec<ZZ_p> y;
                if (index_by_ZZ) {
                x.SetLength(2 * ncol);
                y.SetLength(2 * ncol);
                } else {
                x.SetLength(ncol);
                y.SetLength(ncol);
                }
                for (int j = 0; j < ncol; j++) {
                x[j] = j + 1;
                y[j] = table_cache[cid][i][j];
                if (index_by_ZZ) {
                    x[j + ncol] = x[j] + conv<long>(primes[table_field_index[cid]]);
                    y[j + ncol] = table_cache[cid][i][j];
                }
                }

                lagrange_interp(lagrange_cache[cid][i], x, y);
            }
            }
        """
        pass
    
    def _construct_cache(self: 'Client'):
        self._construct_lagrange_cache()
    
    def client_connect(self: 'Client', other: 'Client'):
        # Temp local solution
        self.__other_clients[other.pid] = other
    
    def client_disconnect(self: 'Client', other: 'Client'):
        # Temp local solution:
        del self.__other_clients[other.pid]

    def call(self: 'Client', secure_fn: Callable, context_id: int) -> Any:
        return secure_fn(self, context_id, *self.__context[context_id])
    
    def append_to_context(self: 'Client', context_id: int, shared: Any):
        self.__context[context_id].append(shared)
    
    def prune_context(self: 'Client', context_id: int):
        if context_id in self.__context:
            del self.__context[context_id]
    
    def get_shared(self: 'Client', context_id: int, secrets: list, transform: Callable) -> Any:
        return transform([self.__context[context_id][i] for i in secrets])
    
    def get_param(self: 'Client', context_id: int, param_index: int) -> Any:
        return self.__context[context_id][param_index]

    def get_counter_client(self: 'Client') -> 'Client':
        if self.pid == CP0:
            raise ValueError('Preprocess client has no counter client.')
        
        return self.__other_clients[int(not self.pid)]
    
    def get_other_clients(self: 'Client') -> list:
        return list(self.__other_clients.values())

    def reconstruct_secret(self: 'Client', context_id: int, secrets: list, transform: Callable) -> Any:
        self_shared = self.get_shared(context_id, secrets, transform)
        other_shared = self.get_counter_client().get_shared(context_id, secrets, transform)

        return self_shared + other_shared
    
    def load_shared_from_path(self: 'Client', context_id: Any, data_path: str) -> list:
        raise NotImplementedError()
