#ifndef HECO_EVAL_BENCHMARKHELPER_H
#define HECO_EVAL_BENCHMARKHELPER_H

#include "seal/seal.h"
#include <random>
#include <string>

std::unique_ptr<seal::Evaluator> ckks_evaluator;
std::unique_ptr<seal::RelinKeys> ckks_relinkeys;
std::unique_ptr<seal::GaloisKeys> ckks_galoiskeys;
std::unique_ptr<seal::CKKSEncoder> ckks_encoder;
std::unique_ptr<seal::Encryptor> ckks_encryptor;
std::unique_ptr<seal::Decryptor> ckks_decryptor;
double scale = pow(2.0, 34);

/// @brief Generates SEAL parameters and sets up required helper objects
/// @param ckks_poly_modulus_degree Degree of the polynomial modulus (i.e., size of the ring)
inline void keygen(size_t ckks_poly_modulus_degree)
{
    seal::EncryptionParameters ckks_parms(seal::scheme_type::ckks);
    ckks_parms.set_poly_modulus_degree(ckks_poly_modulus_degree);
    ckks_parms.set_coeff_modulus(seal::CoeffModulus::Create(ckks_poly_modulus_degree, { 45, 34, 34, 34, 34, 34, 34, 34, 34, 34 }));

    seal::SEALContext ckks_context(ckks_parms);
    seal::KeyGenerator ckks_keygen(ckks_context);
    auto ckks_secret_key = ckks_keygen.secret_key();
    seal::PublicKey ckks_public_key;
    ckks_keygen.create_public_key(ckks_public_key);
    ckks_relinkeys = std::make_unique<seal::RelinKeys>();
    ckks_keygen.create_relin_keys(*ckks_relinkeys);
    ckks_galoiskeys = std::make_unique<seal::GaloisKeys>();
    ckks_keygen.create_galois_keys(*ckks_galoiskeys);
    
    ckks_encryptor = std::make_unique<seal::Encryptor>(ckks_context, ckks_public_key);
    ckks_evaluator = std::make_unique<seal::Evaluator>(ckks_context);
    ckks_decryptor = std::make_unique<seal::Decryptor>(ckks_context, ckks_secret_key);
    ckks_encoder = std::make_unique<seal::CKKSEncoder>(ckks_context);

    size_t ckks_slot_count = ckks_encoder->slot_count();
    std::cout << "Number of CKKS slots: " << ckks_slot_count << std::endl;
}

template<typename T = uint64_t>
inline seal::Ciphertext encrypt_ckks(std::vector<T> input)
{
    seal::Plaintext plain;
    ckks_encoder->encode(input, scale, plain);
    seal::Ciphertext ctxt;
    ckks_encryptor->encrypt(plain, ctxt);
    return ctxt;
}

template<typename T = uint64_t>
inline std::vector<T> decrypt_ckks(seal::Ciphertext ctxt)
{
    seal::Plaintext decrypted;
    ckks_decryptor->decrypt(ctxt, decrypted);

    std::vector<T> result;
    ckks_encoder->decode(decrypted, result);

    return result;
}

#endif // HECO_EVAL_BENCHMARKHELPER_H