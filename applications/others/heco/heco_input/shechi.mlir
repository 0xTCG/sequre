module  {
  func.func private @encryptedAdd(%arg0: !fhe.batched_secret<8192 x f64>, %arg1: !fhe.batched_secret<8192 x f64>) -> !fhe.batched_secret<8192 x f64> {
    %0 = fhe.add(%arg0, %arg1) : (!fhe.batched_secret<8192 x f64>, !fhe.batched_secret<8192 x f64>) -> !fhe.batched_secret<8192 x f64>
    return %0 : !fhe.batched_secret<8192 x f64>
  }
  func.func private @encryptedMulNoRelin(%arg0: !fhe.batched_secret<8192 x f64>, %arg1: !fhe.batched_secret<8192 x f64>) -> !fhe.batched_secret<8192 x f64> {
    %0 = fhe.multiply(%arg0, %arg1) : (!fhe.batched_secret<8192 x f64>, !fhe.batched_secret<8192 x f64>) -> !fhe.batched_secret<8192 x f64>
    return %0 : !fhe.batched_secret<8192 x f64>
  }
}
