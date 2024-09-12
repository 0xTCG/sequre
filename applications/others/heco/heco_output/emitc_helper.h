#ifndef HECO_OUTPUT_EMITC_HELPER_H
#define HECO_OUTPUT_EMITC_HELPER_H

#include "seal/seal.h"

inline void insert(std::vector<seal::Ciphertext> &v, seal::Ciphertext &c)
{
    v.push_back(c);
}

inline seal::Plaintext evaluator_encode(int16_t value)
{
    seal::Plaintext plain;
    std::vector<double> vector(8192, double(value));
    ckks_encoder->encode(vector, scale, plain);
    return plain;
}

inline seal::Ciphertext evaluator_multiply(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    if (a.size() > 2)
        ckks_evaluator->relinearize_inplace(a, *ckks_relinkeys);
    if (b.size() > 2)
        ckks_evaluator->relinearize_inplace(b, *ckks_relinkeys);
    if (&a == &b)
        ckks_evaluator->square(a, result);
    else
        ckks_evaluator->multiply(a, b, result);

    return result;
}

inline seal::Ciphertext evaluator_multiply_plain(seal::Ciphertext &a, seal::Plaintext &b)
{
    seal::Ciphertext result;
    ckks_evaluator->multiply_plain(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_multiply_many(std::vector<seal::Ciphertext> &as, seal::RelinKeys &rlk)
{
    seal::Ciphertext result;
    ckks_evaluator->multiply_many(as, rlk, result);
    return result;
}

inline seal::Ciphertext evaluator_add(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    ckks_evaluator->add(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_sub(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    ckks_evaluator->sub(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_add_many(std::vector<seal::Ciphertext> &as)
{
    seal::Ciphertext result;
    ckks_evaluator->add_many(as, result);
    return result;
}

inline seal::Ciphertext evaluator_relinearize(seal::Ciphertext &a, const seal::RelinKeys &b)
{
    seal::Ciphertext result;
    ckks_evaluator->relinearize(a, b, result);
    return result;
}

inline seal::Ciphertext evaluator_modswitch_to(seal::Ciphertext &a, seal::Ciphertext &b)
{
    seal::Ciphertext result;
    ckks_evaluator->mod_switch_to(a, b.parms_id(), result);
    return result;
}

inline seal::Ciphertext evaluator_rotate(seal::Ciphertext &a, int i)
{
    seal::Ciphertext result;
    if (a.size() > 2)
        ckks_evaluator->relinearize_inplace(a, *ckks_relinkeys);
    ckks_evaluator->rotate_rows(a, i, *ckks_galoiskeys, result);
    return result;
}
#endif // HECO_OUTPUT_EMITC_HELPER_H