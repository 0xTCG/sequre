// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "examples.h"
#include <chrono>
#include <vector>
#include <random>


using namespace std;
using namespace seal;

void example_ckks_basics()
{
    print_example_banner("Example: CKKS Basics");

    /*
    In this example we demonstrate evaluating a polynomial function

        PI*x^3 + 0.4*x + 1

    on encrypted floating-point input data x for a set of 4096 equidistant points
    in the interval [0, 1]. This example demonstrates many of the main features
    of the CKKS scheme, but also the challenges in using it.

    We start by setting up the CKKS scheme.
    */
    EncryptionParameters parms(scheme_type::ckks);

    /*
    We saw in `2_encoders.cpp' that multiplication in CKKS causes scales
    in ciphertexts to grow. The scale of any ciphertext must not get too close
    to the total size of coeff_modulus, or else the ciphertext simply runs out of
    room to store the scaled-up plaintext. The CKKS scheme provides a `rescale'
    functionality that can reduce the scale, and stabilize the scale expansion.

    Rescaling is a kind of modulus switch operation (recall `3_levels.cpp').
    As modulus switching, it removes the last of the primes from coeff_modulus,
    but as a side-effect it scales down the ciphertext by the removed prime.
    Usually we want to have perfect control over how the scales are changed,
    which is why for the CKKS scheme it is more common to use carefully selected
    primes for the coeff_modulus.

    More precisely, suppose that the scale in a CKKS ciphertext is S, and the
    last prime in the current coeff_modulus (for the ciphertext) is P. Rescaling
    to the next level changes the scale to S/P, and removes the prime P from the
    coeff_modulus, as usual in modulus switching. The number of primes limits
    how many rescalings can be done, and thus limits the multiplicative depth of
    the computation.

    It is possible to choose the initial scale freely. One good strategy can be
    to is to set the initial scale S and primes P_i in the coeff_modulus to be
    very close to each other. If ciphertexts have scale S before multiplication,
    they have scale S^2 after multiplication, and S^2/P_i after rescaling. If all
    P_i are close to S, then S^2/P_i is close to S again. This way we stabilize the
    scales to be close to S throughout the computation. Generally, for a circuit
    of depth D, we need to rescale D times, i.e., we need to be able to remove D
    primes from the coefficient modulus. Once we have only one prime left in the
    coeff_modulus, the remaining prime must be larger than S by a few bits to
    preserve the pre-decimal-point value of the plaintext.

    Therefore, a generally good strategy is to choose parameters for the CKKS
    scheme as follows:

        (1) Choose a 60-bit prime as the first prime in coeff_modulus. This will
            give the highest precision when decrypting;
        (2) Choose another 60-bit prime as the last element of coeff_modulus, as
            this will be used as the special prime and should be as large as the
            largest of the other primes;
        (3) Choose the intermediate primes to be close to each other.

    We use CoeffModulus::Create to generate primes of the appropriate size. Note
    that our coeff_modulus is 200 bits total, which is below the bound for our
    poly_modulus_degree: CoeffModulus::MaxBitCount(8192) returns 218.
    */

    // changed for compariosns with Sequre
    size_t poly_modulus_degree = 16384;
    parms.set_poly_modulus_degree(poly_modulus_degree);
    parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, { 45, 34, 34, 34, 34, 34, 34, 34, 34, 34 }));

    /*
    We choose the initial scale to be 2^40. At the last level, this leaves us
    60-40=20 bits of precision before the decimal point, and enough (roughly
    10-20 bits) of precision after the decimal point. Since our intermediate
    primes are 40 bits (in fact, they are very close to 2^40), we can achieve
    scale stabilization as described above.
    */
    double scale = pow(2.0, 34);

    SEALContext context(parms);
    print_parameters(context);
    cout << endl;

    KeyGenerator keygen(context);
    auto secret_key = keygen.secret_key();
    PublicKey public_key;
    keygen.create_public_key(public_key);
    RelinKeys relin_keys;
    keygen.create_relin_keys(relin_keys);
    GaloisKeys gal_keys;
    keygen.create_galois_keys(gal_keys);
    Encryptor encryptor(context, public_key);
    Evaluator evaluator(context);
    Decryptor decryptor(context, secret_key);

    CKKSEncoder encoder(context);
    size_t slot_count = encoder.slot_count();
    cout << "Number of slots: " << slot_count << endl;

    vector<double> input;
    input.reserve(slot_count);
    double curr_point = 0;
    double step_size = 1.0 / (static_cast<double>(slot_count) - 1);
    for (size_t i = 0; i < slot_count; i++)
    {
        input.push_back(curr_point);
        curr_point += step_size;
    }
    cout << "Input vector: " << endl;
    print_vector(input, 3, 7);

////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// MICROBENCHMARKS
////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // our own tests
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(-100.0, 100.0); // Adjust range as needed

    // Create a vector to store the random values
    std::vector<double> randomValues(8192);
    stringstream data_stream;

    // Generate random values and store them in the vector
    for (int i = 0; i < 8192; ++i) {
        randomValues[i] = dist(gen);
    }

    // Variable to store the total time taken
        double totalTimeEnc = 0.0;
        double totalTimePlainAdd = 0.0;
        double totalTimeAdd = 0.0;
        double totalTimePlainMult = 0.0;
        double totalTimeMult = 0.0;
        double totalTimeRotate = 0.0;
        double totalTimeDec = 0.0;

        int run_count = 100;
        
        // Repeat the process multiple times
        for (int iter = 0; iter < run_count; ++iter) {
            Plaintext plain_vec1;
            Ciphertext encrypted_result;
            Ciphertext vec1_encrypted;
            
            // encryption
            auto start = std::chrono::high_resolution_clock::now();
            encoder.encode(randomValues, scale, plain_vec1);
            encryptor.encrypt(plain_vec1, vec1_encrypted);
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;
            totalTimeEnc += duration.count();

            // plain add
            // only this in latest manuscript version
            start = std::chrono::high_resolution_clock::now();
            evaluator.add_plain(vec1_encrypted, plain_vec1, encrypted_result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimePlainAdd += duration.count();

            // add
            start = std::chrono::high_resolution_clock::now();
            evaluator.add(vec1_encrypted, vec1_encrypted, encrypted_result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimeAdd += duration.count();

            // plain mult
            // only this in latest manuscript version
            start = std::chrono::high_resolution_clock::now();
            evaluator.multiply_plain(vec1_encrypted, plain_vec1,encrypted_result);
            evaluator.rescale_to_next_inplace(encrypted_result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimePlainMult += duration.count();

            // mult
            start = std::chrono::high_resolution_clock::now();
            evaluator.multiply(vec1_encrypted, vec1_encrypted,encrypted_result);
            evaluator.relinearize_inplace(encrypted_result, relin_keys);
            evaluator.rescale_to_next_inplace(encrypted_result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimeMult += duration.count();

            // rotate
            start = std::chrono::high_resolution_clock::now();
            evaluator.rotate_vector(vec1_encrypted, 2, gal_keys, encrypted_result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimeRotate += duration.count();

            // decrypt
            start = std::chrono::high_resolution_clock::now();
            decryptor.decrypt(encrypted_result, plain_vec1);
            vector<double> result;
            encoder.decode(plain_vec1, result);
            end = std::chrono::high_resolution_clock::now();
            duration = end - start;
            totalTimeDec += duration.count();
        }

        // Calculate the average time taken
        double averageTimeEnc = totalTimeEnc / run_count;
        double averageTimePlainAdd = totalTimePlainAdd / run_count;
        double averageTimeAdd = totalTimeAdd / run_count;
        double averageTimePlainMult = totalTimePlainMult / run_count;
        double averageTimeMult = totalTimeMult / run_count;
        double averageTimeRotate = totalTimeRotate / run_count;
        double averageTimeDec = totalTimeDec / run_count;

        std::cout << "Average time for encryption: " << averageTimeEnc << " seconds" << std::endl;
        std::cout << "Average time for plain addition: " << averageTimePlainAdd << " seconds" << std::endl;
        std::cout << "Average time for addition: " << averageTimeAdd << " seconds" << std::endl;
        std::cout << "Average time for plain multiplication: " << averageTimePlainMult << " seconds" << std::endl;
        std::cout << "Average time for multiplication: " << averageTimeMult << " seconds" << std::endl;
        std::cout << "Average time for rotation: " << averageTimeRotate << " seconds" << std::endl;
        std::cout << "Average time for decryption: " << averageTimeDec << " seconds" << std::endl;
        
}
