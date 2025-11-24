// SPDX-License-Identifier: MIT

#include <stdlib.h>
#include <oqs/oqs.h>
#include <oqs/kem_ntru_plus.h>

// [중요] 최적화된 새 폴더의 api.h 포함 (경로: normalization)
#include "KpqClean_ver2_NTRU_PLUS_KEM576_clean_montgomery-batch-normalization/api.h"

#if defined(OQS_ENABLE_KEM_ntru_plus_kem576)

// 1. 실제 구현체 함수들을 extern으로 직접 선언 (매크로 의존 X)
extern int kpqclean_ntruplus576_opt_crypto_kem_keypair(uint8_t *pk, uint8_t *sk);
extern int kpqclean_ntruplus576_opt_crypto_kem_enc(uint8_t *ct, uint8_t *ss, const uint8_t *pk);
extern int kpqclean_ntruplus576_opt_crypto_kem_dec(uint8_t *ss, const uint8_t *ct, const uint8_t *sk);

// 2. OQS 래퍼 함수 원형 선언 (Forward Declaration)
OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_keypair(uint8_t *public_key, uint8_t *secret_key);
OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_encaps(uint8_t *ciphertext, uint8_t *shared_secret, const uint8_t *public_key);
OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_decaps(uint8_t *shared_secret, const uint8_t *ciphertext, const uint8_t *secret_key);
OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_keypair_derand(uint8_t *public_key, uint8_t *secret_key, const uint8_t *seed);

// 3. 생성자 함수
OQS_KEM *OQS_KEM_ntru_plus_kem576_opt_new(void) {
    OQS_KEM *kem = OQS_MEM_malloc(sizeof(OQS_KEM));
    if (kem == NULL) {
        return NULL;
    }
    kem->method_name = "NTRU-Plus-KEM-576-Opt";
    kem->alg_version = "Montgomery-Batch-Normalization-v1";
    kem->claimed_nist_level = 1;
    kem->ind_cca = true;

    kem->length_public_key = OQS_KEM_ntru_plus_kem576_length_public_key;
    kem->length_secret_key = OQS_KEM_ntru_plus_kem576_length_secret_key;
    kem->length_ciphertext = OQS_KEM_ntru_plus_kem576_length_ciphertext;
    kem->length_shared_secret = OQS_KEM_ntru_plus_kem576_length_shared_secret;
    kem->length_keypair_seed = OQS_KEM_ntru_plus_kem576_length_keypair_seed;

    kem->keypair = OQS_KEM_ntru_plus_kem576_opt_keypair;
    kem->keypair_derand = OQS_KEM_ntru_plus_kem576_opt_keypair_derand;
    kem->encaps = OQS_KEM_ntru_plus_kem576_opt_encaps;
    kem->decaps = OQS_KEM_ntru_plus_kem576_opt_decaps;

    return kem;
}

// 4. 래퍼 함수 구현 (extern 함수 호출)
OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_keypair(uint8_t *public_key, uint8_t *secret_key) {
    return (OQS_STATUS) kpqclean_ntruplus576_opt_crypto_kem_keypair(public_key, secret_key);
}

OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_encaps(uint8_t *ciphertext, uint8_t *shared_secret, const uint8_t *public_key) {
    return (OQS_STATUS) kpqclean_ntruplus576_opt_crypto_kem_enc(ciphertext, shared_secret, public_key);
}

OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_decaps(uint8_t *shared_secret, const uint8_t *ciphertext, const uint8_t *secret_key) {
    return (OQS_STATUS) kpqclean_ntruplus576_opt_crypto_kem_dec(shared_secret, ciphertext, secret_key);
}

OQS_API OQS_STATUS OQS_KEM_ntru_plus_kem576_opt_keypair_derand(uint8_t *public_key, uint8_t *secret_key, const uint8_t *seed) {
    (void)public_key; (void)secret_key; (void)seed;
    return OQS_ERROR;
}

#endif
