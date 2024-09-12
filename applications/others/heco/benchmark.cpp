#include "seal/seal.h"
#include <cassert>
#include <memory>
#include "../util/BenchmarkHelper.h"
#include "../util/MultiTimer.h"

////////////////////////////////
// Include compiled code here //
////////////////////////////////
#include "heco_output/emitc_helper.h"
// ^- defines the non-member versions of evluator.*
#include "heco_output/shechi.cpp"

////////////////////////////////
//    BENCHMARK FUNCTIONS     //
////////////////////////////////

// Number of iterations for the benchmark
// set to 10 for actual evaluation
#define ITER_COUNT 10

void microbench(size_t size = 8192)
{
    // Generate Input vectors (technically, they should be boolean, but no impact on performance)
    // std::vector<double> x(size);
    // for (size_t i = 0; i != size; ++i) x.push_back(rand());
    // std::vector<double> y(size);
    // for (size_t i = 0; i != size; ++i) y.push_back(rand());
    std::vector<double> input;
    input.reserve(size);
    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(size) - 1);
    for (size_t i = 0; i < size; i++) {
        input.push_back(curr_point);
        curr_point += step_size;
    }

    MultiTimer timer = MultiTimer();
    std::cout << "Running BoxBlurBench (" << size << ") HECO Version: " << std::endl;
    for (int i = 0; i < ITER_COUNT; ++i)
    {
        // KeyGen
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (KeyGen)            " << std::endl;
        auto keygenTimer = timer.startTimer();
        keygen(size * 2);
        timer.stopTimer(keygenTimer);

        // Encryption
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Encryption)            " << std::endl;
        auto encodeTimer = timer.startTimer();
        seal::Plaintext plain;
        ckks_encoder->encode(input, scale, plain);
        timer.stopTimer(encodeTimer);
        auto encryptTimer = timer.startTimer();
        seal::Ciphertext ctxt_x;
        ckks_encryptor->encrypt(plain, ctxt_x);
        timer.stopTimer(encryptTimer);
        auto ctxt_y = encrypt_ckks<double>(input);

        seal::Ciphertext evaluated;
        seal::Plaintext plain_vec = evaluator_encode(3);
        
        // Add plain !!HECO MLIR does not work
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Plain addition)            " << std::endl;
        auto addPlainTimer = timer.startTimer();
        ckks_evaluator->add_plain(ctxt_x, plain_vec, evaluated);
        timer.stopTimer(addPlainTimer);
        
        // Add
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Addition)            " << std::endl;
        auto addTimer = timer.startTimer();
        evaluated = encryptedAdd(ctxt_x, ctxt_y);
        timer.stopTimer(addTimer);

        // Mul w/o relin
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Multiplication w/o relinearization)            " << std::endl;
        auto mulNoRelinTimer = timer.startTimer();
        evaluated = encryptedMulNoRelin(ctxt_x, ctxt_y);
        timer.stopTimer(mulNoRelinTimer);

        // Mul w/ relin
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Multiplication w/ relinearization)            " << std::endl;
        auto mulRelinTimer = timer.startTimer();
        evaluated = encryptedMulNoRelin(evaluated, ctxt_y);
        timer.stopTimer(mulRelinTimer);

        // Mul plain !!HECO MLIR does not work
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Plain multiplication)            " << std::endl;
        auto mulPlain = timer.startTimer();
        ckks_evaluator->multiply_plain(ctxt_x, plain_vec, evaluated);
        ckks_evaluator->rescale_to_next_inplace(evaluated);
        timer.stopTimer(mulPlain);

        // Mul !!HECO MLIR does not work
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Multiplication)            " << std::endl;
        auto mulTimer = timer.startTimer();
        ckks_evaluator->multiply(ctxt_x, ctxt_y, evaluated);
        ckks_evaluator->relinearize_inplace(evaluated, *ckks_relinkeys);
        ckks_evaluator->rescale_to_next_inplace(evaluated);
        timer.stopTimer(mulTimer);

        // Rotation !!HECO MLIR does not work
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Rotation)            " << std::endl;
        auto rotTimer = timer.startTimer();
        ckks_evaluator->rotate_vector(ctxt_x, 3, *ckks_galoiskeys, evaluated);
        timer.stopTimer(rotTimer);

        // Decryption
        std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << i + 1 << "/" << ITER_COUNT
                    << " (Decryption)            " << std::endl;
        auto decryptTimer = timer.startTimer();
        seal::Plaintext decrypted;
        ckks_decryptor->decrypt(ctxt_x, decrypted);
        timer.stopTimer(decryptTimer);
        auto decodeTimer = timer.startTimer();
        std::vector<double> result;
        ckks_encoder->decode(decrypted, result);
        timer.stopTimer(decodeTimer);

        timer.addIteration();
    }
    timer.printToFile("evaluation/plotting/data/benchmark/microbench" + std::to_string(size) + ".csv");
    std::cout << "\rRunning BoxBlurBench (" << size << ") HECO Version: " << ITER_COUNT << "/" << ITER_COUNT
                << " (DONE)            " << std::endl;
}


int main()
{
    microbench();
}
