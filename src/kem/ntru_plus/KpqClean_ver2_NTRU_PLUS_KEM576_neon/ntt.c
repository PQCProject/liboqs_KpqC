#include "params.h"
#include <oqs/oqsconfig.h>
#include "reduce.h"
#include "ntt.h"

#if defined(OQS_USE_ARM_NEON_INSTRUCTIONS) && (defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define NTRUPLUS_HAVE_NEON 1
#endif

const int16_t zetas[144] = {
	 -147, -1033, -1265,   708,   460,  1265,  -467,   727,
	  556,  1307,  -773,  -161,  1200, -1612,   570,  1529,
	 1135,  -556,  1120,   298,  -822, -1556,   -93,  1463,
	  532,  -377,  -909,    58,  -392,  -450,  1722,  1236,
	 -486,  -491, -1569, -1078,    36,  1289, -1443,  1628,
	 1664,  -725,  -952,    99, -1020,   353,  -599,  1119,
	  592,   839,  1622,   652,  1244,  -783, -1085,  -726,
	  566,  -284, -1369, -1292,   268,  -391,   781,  -172,
	   96, -1172,   211,   737,   473,  -445,  -234,   264,
	-1536,  1467,  -676, -1542,  -170,   635,  -705, -1332,
	 -658,   831, -1712,  1311,  1488,  -881,  1087, -1315,
	 1245,   -75,   791,    -6,  -875,  -697,   -70, -1162,
	  287,  -767,  -945,  1598,  -882,  1261,   206,   654,
	-1421,   -81,   716, -1251,   838, -1300,  1035,  -104,
	  966,  -558,   -61, -1704,   404,  -899,   862, -1593,
	-1460,   -37,  1266,   965, -1584, -1404,  -265,  -942,
	  905,  1195,  -619,   787,   118,   576,   286, -1475,
	 -194,   928,  1229, -1032,  1608,  1111, -1669,   642
};

#if defined(NTRUPLUS_HAVE_NEON)
static inline int16x4_t montgomery_reduce_vec4(int32x4_t a)
{
	int16x4_t a16 = vmovn_s32(a);
	int16x4_t t16 = vmul_n_s16(a16, QINV);
	int32x4_t prod = vmull_s16(t16, vdup_n_s16(NTRUPLUS_Q));
	int32x4_t res = vshrq_n_s32(vsubq_s32(a, prod), 16);
	return vmovn_s32(res);
}

static inline int16x4_t fqmul_vec4_const(int16x4_t a, int16_t zeta)
{
	int32x4_t prod = vmull_s16(a, vdup_n_s16(zeta));
	return montgomery_reduce_vec4(prod);
}

static inline int16x8_t fqmul_vec8_const(int16x8_t a, int16_t zeta)
{
	int16x4_t lo = fqmul_vec4_const(vget_low_s16(a), zeta);
	int16x4_t hi = fqmul_vec4_const(vget_high_s16(a), zeta);
	return vcombine_s16(lo, hi);
}

static inline int16x4_t barrett_reduce_vec4(int16x4_t a)
{
	const int32_t v = ((1 << 26) + NTRUPLUS_Q/2)/NTRUPLUS_Q;
	int32x4_t a32 = vmovl_s16(a);
	int32x4_t t = vaddq_s32(vmulq_n_s32(a32, v), vdupq_n_s32(1 << 25));
	t = vshrq_n_s32(t, 26);
	t = vmulq_n_s32(t, NTRUPLUS_Q);
	return vmovn_s32(vsubq_s32(a32, t));
}

static inline int16x8_t barrett_reduce_vec8(int16x8_t a)
{
	int16x4_t lo = barrett_reduce_vec4(vget_low_s16(a));
	int16x4_t hi = barrett_reduce_vec4(vget_high_s16(a));
	return vcombine_s16(lo, hi);
}
#endif

/*************************************************
* Name:        fqmul
*
* Description: Multiplication followed by Montgomery reduction
*
* Arguments:   - int16_t a: first factor
*              - int16_t b: second factor
*
* Returns 16-bit integer congruent to a*b*R^{-1} mod q
**************************************************/
static int16_t fqmul(int16_t a, int16_t b)
{
	return montgomery_reduce((int32_t)a*b);
}

/*************************************************
* Name:        fqinv
*
* Description: Inversion
*
* Arguments:   - int16_t a: first factor a = x mod q
*
* Returns 16-bit integer congruent to x^{-1} * R^2 mod q
**************************************************/
static int16_t fqinv(int16_t a)
{
	int16_t t1,t2,t3;

	t1 = fqmul(a, a);    //10
	t2 = fqmul(t1, t1);  //100
	t2 = fqmul(t2, t2);  //1000
	t3 = fqmul(t2, t2);  //10000

	t1 = fqmul(t1, t2);  //1010

	t2 = fqmul(t1, t3);  //11010
	t2 = fqmul(t2, t2);  //110100
	t2 = fqmul(t2, a);   //110101

	t1 = fqmul(t1, t2);  //111111

	t2 = fqmul(t2, t2);  //1101010
	t2 = fqmul(t2, t2);  //11010100
	t2 = fqmul(t2, t2);  //110101000
	t2 = fqmul(t2, t2);  //1101010000
	t2 = fqmul(t2, t2);  //11010100000
	t2 = fqmul(t2, t2);  //110101000000
	t2 = fqmul(t2, t1);  //110101111111

	return t2;
}

/*************************************************
* Name:        ntt
*
* Description: number-theoretic transform (NTT) in Rq.
*
* Arguments:   - int16_t r[NTRUPLUS_N]: pointer to output vector of elements of Zq
*              - int16_t a[NTRUPLUS_N]: pointer to input vector of elements of Zq
**************************************************/
void ntt(int16_t r[NTRUPLUS_N], const int16_t a[NTRUPLUS_N])
{
#if defined(NTRUPLUS_HAVE_NEON)
	int k = 1;
	int16_t zeta1 = zetas[k++];

	for(int i = 0; i < NTRUPLUS_N/2; i += 8)
	{
		int16x8_t a0 = vld1q_s16(a + i);
		int16x8_t a1 = vld1q_s16(a + i + NTRUPLUS_N/2);
		int16x8_t t1 = fqmul_vec8_const(a1, zeta1);

		vst1q_s16(r + i + NTRUPLUS_N/2, vsubq_s16(vaddq_s16(a0, a1), t1));
		vst1q_s16(r + i, vaddq_s16(a0, t1));
	}

	for(int step = NTRUPLUS_N/6; step >= 32; step = step/3)
	{
		for(int start = 0; start < NTRUPLUS_N; start += 3*step)
		{
			int16_t zeta_a = zetas[k++];
			int16_t zeta_b = zetas[k++];

			for(int i = start; i < start + step; i += 8)
			{
				int16x8_t r0 = vld1q_s16(r + i);
				int16x8_t r1 = vld1q_s16(r + i + step);
				int16x8_t r2 = vld1q_s16(r + i + 2*step);
				int16x8_t t1 = fqmul_vec8_const(r1, zeta_a);
				int16x8_t t2 = fqmul_vec8_const(r2, zeta_b);
				int16x8_t diff = vsubq_s16(t1, t2);
				int16x8_t t3 = fqmul_vec8_const(diff, -886);

				vst1q_s16(r + i + 2*step, vsubq_s16(vsubq_s16(r0, t1), t3));
				vst1q_s16(r + i +   step, vaddq_s16(vsubq_s16(r0, t2), t3));
				vst1q_s16(r + i, vaddq_s16(r0, vaddq_s16(t1, t2)));
			}
		}
	}

	for(int step = 16; step >= 4; step >>= 1)
	{
		for(int start = 0; start < NTRUPLUS_N; start += (step << 1))
		{
			int16_t zeta = zetas[k++];

			if(step >= 8)
			{
				for(int i = start; i < start + step; i += 8)
				{
					int16x8_t r0 = vld1q_s16(r + i);
					int16x8_t r1 = vld1q_s16(r + i + step);
					int16x8_t t1 = fqmul_vec8_const(r1, zeta);
					int16x8_t sum = vaddq_s16(r0, t1);
					int16x8_t diff = vsubq_s16(r0, t1);

					vst1q_s16(r + i, barrett_reduce_vec8(sum));
					vst1q_s16(r + i + step, barrett_reduce_vec8(diff));
				}
			}
			else
			{
				for(int i = start; i < start + step; i += 4)
				{
					int16x4_t r0 = vld1_s16(r + i);
					int16x4_t r1 = vld1_s16(r + i + step);
					int16x4_t t1 = fqmul_vec4_const(r1, zeta);
					int16x4_t sum = vadd_s16(r0, t1);
					int16x4_t diff = vsub_s16(r0, t1);

					vst1_s16(r + i, barrett_reduce_vec4(sum));
					vst1_s16(r + i + step, barrett_reduce_vec4(diff));
				}
			}
		}
	}
#else
	int16_t t1,t2,t3;
	int16_t zeta1,zeta2;
	
	int k = 1;

	zeta1 = zetas[k++];

	for(int i = 0; i < NTRUPLUS_N/2; i++)
	{
		t1 = fqmul(zeta1, a[i + NTRUPLUS_N/2]);

		r[i + NTRUPLUS_N/2] = a[i] + a[i + NTRUPLUS_N/2] - t1;
		r[i               ] = a[i]                       + t1;
	}

	for(int step = NTRUPLUS_N/6; step >= 32; step = step/3)
	{
		for(int start = 0; start < NTRUPLUS_N; start += 3*step)
		{
			zeta1 = zetas[k++];
			zeta2 = zetas[k++];

			for(int i = start; i < start + step; i++)
			{
				t1 = fqmul(zeta1, r[i +   step]);
				t2 = fqmul(zeta2, r[i + 2*step]);
				t3 = fqmul(-886, t1 - t2);

				r[i + 2*step] = r[i] - t1 - t3;
				r[i +   step] = r[i] - t2 + t3;
				r[i         ] = r[i] + t1 + t2;
			}		
		}
	}

	for(int step = 16; step >= 4; step >>= 1)
	{
		for(int start = 0; start < NTRUPLUS_N; start += (step << 1))
		{
			zeta1 = zetas[k++];

			for(int i = start; i < start + step; i++)
			{
				t1 = fqmul(zeta1, r[i + step]);
				
				r[i + step] = barrett_reduce(r[i] - t1);
				r[i       ] = barrett_reduce(r[i] + t1);
			}
		}
	}
#endif
}

/*************************************************
* Name:        invntt
*
* Description: inverse number-theoretic transform in Rq and
*              multiplication by Montgomery factor R = 2^16.
*
* Arguments:   - int16_t r[NTRUPLUS_N]: pointer to output vector of elements of Zq
*              - int16_t a[NTRUPLUS_N]: pointer to input vector of elements of Zq
**************************************************/
void invntt(int16_t r[NTRUPLUS_N], const int16_t a[NTRUPLUS_N])
{
#if defined(NTRUPLUS_HAVE_NEON)
	int k = 143;

	for(int i = 0; i < NTRUPLUS_N; i += 8)
	{
		int16x8_t v = vld1q_s16(a + i);
		vst1q_s16(r + i, v);
	}

	for(int step = 4; step <= 16; step <<= 1)
	{
		for(int start = 0; start < NTRUPLUS_N; start += (step << 1))
		{
			int16_t zeta = zetas[k--];

			if(step >= 8)
			{
				for(int i = start; i < start + step; i += 8)
				{
					int16x8_t r0 = vld1q_s16(r + i);
					int16x8_t r1 = vld1q_s16(r + i + step);
					int16x8_t sum = vaddq_s16(r0, r1);
					int16x8_t diff = vsubq_s16(r1, r0);

					vst1q_s16(r + i, barrett_reduce_vec8(sum));
					vst1q_s16(r + i + step, fqmul_vec8_const(diff, zeta));
				}
			}
			else
			{
				for(int i = start; i < start + step; i += 4)
				{
					int16x4_t r0 = vld1_s16(r + i);
					int16x4_t r1 = vld1_s16(r + i + step);
					int16x4_t sum = vadd_s16(r0, r1);
					int16x4_t diff = vsub_s16(r1, r0);

					vst1_s16(r + i, barrett_reduce_vec4(sum));
					vst1_s16(r + i + step, fqmul_vec4_const(diff, zeta));
				}
			}
		}
	}

	for(int step = 32; step <= NTRUPLUS_N/6; step = 3*step)
	{
		for(int start = 0; start < NTRUPLUS_N; start += 3*step)
		{
			int16_t zeta2 = zetas[k--];
			int16_t zeta1 = zetas[k--];

			for(int i = start; i < start + step; i += 8)
			{
				int16x8_t r0 = vld1q_s16(r + i);
				int16x8_t r1 = vld1q_s16(r + i + step);
				int16x8_t r2 = vld1q_s16(r + i + 2*step);
				int16x8_t diff1 = vsubq_s16(r1, r0);
				int16x8_t t1 = fqmul_vec8_const(diff1, -886);
				int16x8_t t2 = fqmul_vec8_const(vaddq_s16(vsubq_s16(r2, r0), t1), zeta1);
				int16x8_t t3 = fqmul_vec8_const(vsubq_s16(vsubq_s16(r2, r1), t1), zeta2);

				vst1q_s16(r + i, barrett_reduce_vec8(vaddq_s16(vaddq_s16(r0, r1), r2)));
				vst1q_s16(r + i + step, t2);
				vst1q_s16(r + i + 2*step, t3);
			}
		}
	}

	for(int i = 0; i < NTRUPLUS_N/2; i += 8)
	{
		int16x8_t r0 = vld1q_s16(r + i);
		int16x8_t r1 = vld1q_s16(r + i + NTRUPLUS_N/2);
		int16x8_t t1 = vaddq_s16(r0, r1);
		int16x8_t diff = vsubq_s16(r0, r1);
		int16x8_t t2 = fqmul_vec8_const(diff, -1665);
		int16x8_t out0 = fqmul_vec8_const(vsubq_s16(t1, t2), -66);
		int16x8_t out1 = fqmul_vec8_const(t2, -132);

		vst1q_s16(r + i, out0);
		vst1q_s16(r + i + NTRUPLUS_N/2, out1);
	}
#else
	int16_t t1, t2, t3;
	int16_t zeta1, zeta2;
	int k = 143;

	for(int i = 0; i < NTRUPLUS_N; i++)
	{
		r[i] = a[i];
	}

	for(int step = 4; step <= 16; step <<= 1)
	{
		for(int start = 0; start < NTRUPLUS_N; start += (step << 1))
		{
			zeta1 = zetas[k--];

			for(int i = start; i < start + step; i++)
			{
				t1 = r[i + step];

				r[i + step] = fqmul(zeta1,  t1 - r[i]);
				r[i       ] = barrett_reduce(r[i] + t1);
			}
		}
	}

	for(int step = 32; step <= NTRUPLUS_N/6; step = 3*step)
	{
		for(int start = 0; start < NTRUPLUS_N; start += 3*step)
		{
			zeta2 = zetas[k--];
			zeta1 = zetas[k--];

			for(int i = start; i < start + step; i++)
			{
				t1 = fqmul(-886,  r[i +   step] - r[i]);
				t2 = fqmul(zeta1, r[i + 2*step] - r[i]        + t1);
				t3 = fqmul(zeta2, r[i + 2*step] - r[i + step] - t1);

				r[i         ] = barrett_reduce(r[i] + r[i + step] + r[i + 2*step]);
				r[i +   step] = t2;			
				r[i + 2*step] = t3;
			}
		}
	}

	for(int i = 0; i < NTRUPLUS_N/2; i++)
	{
		t1 = r[i] + r[i + NTRUPLUS_N/2];
		t2 = fqmul(-1665, r[i] - r[i + NTRUPLUS_N/2]);

		r[i               ] = fqmul(-66, t1 - t2);
		r[i + NTRUPLUS_N/2] = fqmul(-132, t2);
	}
#endif
}

/*************************************************
* Name:        baseinv
*
* Description: Inversion of polynomial in Zq[X]/(X^4-zeta)
*              used for inversion of element in Rq in NTT domain
*
* Arguments:   - int16_t r[4]: pointer to the output polynomial
*              - const int16_t a[4]: pointer to the input polynomial
*              - int16_t zeta: integer defining the reduction polynomial
**************************************************/
int baseinv(int16_t r[4], const int16_t a[4], int16_t zeta)
{
	int16_t t0, t1, t2, t3;
	
	t0 = montgomery_reduce(a[2]*a[2] - (a[1]*a[3] << 1));
	t1 = montgomery_reduce(a[3]*a[3]);
	t0 = montgomery_reduce(a[0]*a[0] + t0*zeta);
	t1 = montgomery_reduce(a[1]*a[1] + t1*zeta - ((a[0]*a[2]) << 1));
	t2 = montgomery_reduce(t1*zeta);
	
	t3 = montgomery_reduce(t0*t0 - t1*t2);

	if(t3 == 0) return 1;

	r[0] = montgomery_reduce(a[0]*t0 + a[2]*t2);
	r[1] = montgomery_reduce(a[3]*t2 + a[1]*t0);
	r[2] = montgomery_reduce(a[2]*t0 + a[0]*t1);
	r[3] = montgomery_reduce(a[1]*t1 + a[3]*t0);

	t3 = fqinv(t3);

	r[0] =  montgomery_reduce(r[0]*t3);
	r[1] = -montgomery_reduce(r[1]*t3);
	r[2] =  montgomery_reduce(r[2]*t3);
	r[3] = -montgomery_reduce(r[3]*t3);

	return 0;
}

/*************************************************
* Name:        basemul
*
* Description: Multiplication of polynomials in Zq[X]/(X^4-zeta)
*              used for multiplication of elements in Rq in NTT domain
*
* Arguments:   - int16_t r[4]: pointer to the output polynomial
*              - const int16_t a[4]: pointer to the first factor
*              - const int16_t b[4]: pointer to the second factor
*              - int16_t zeta: integer defining the reduction polynomial
**************************************************/
void basemul(int16_t r[4], const int16_t a[4], const int16_t b[4], int16_t zeta)
{
	r[0] = montgomery_reduce(a[1]*b[3]+a[2]*b[2]+a[3]*b[1]);
	r[1] = montgomery_reduce(a[2]*b[3]+a[3]*b[2]);
	r[2] = montgomery_reduce(a[3]*b[3]);

	r[0] = montgomery_reduce(r[0]*zeta+a[0]*b[0]);
	r[1] = montgomery_reduce(r[1]*zeta+a[0]*b[1]+a[1]*b[0]);
	r[2] = montgomery_reduce(r[2]*zeta+a[0]*b[2]+a[1]*b[1]+a[2]*b[0]);
	r[3] = montgomery_reduce(a[0]*b[3]+a[1]*b[2]+a[2]*b[1]+a[3]*b[0]);
}

/*************************************************
* Name:        basemul_add
*
* Description: Multiplication then addition of polynomials in Zq[X]/(X^4-zeta)
*              used for multiplication of elements in Rq in NTT domain
*
* Arguments:   - int16_t c[4]: pointer to the output polynomial
*              - const int16_t a[4]: pointer to the first factor
*              - const int16_t b[4]: pointer to the second factor
*              - const int16_t c[4]: pointer to the third factor
*              - int16_t zeta: integer defining the reduction polynomial
**************************************************/
void basemul_add(int16_t r[4], const int16_t a[4], const int16_t b[4], const int16_t c[4], int16_t zeta)
{
	r[0] = montgomery_reduce(a[1]*b[3]+a[2]*b[2]+a[3]*b[1]);
	r[1] = montgomery_reduce(a[2]*b[3]+a[3]*b[2]);
	r[2] = montgomery_reduce(a[3]*b[3]);

	r[0] = montgomery_reduce(c[0]*(-147)+r[0]*zeta+a[0]*b[0]);
	r[1] = montgomery_reduce(c[1]*(-147)+r[1]*zeta+a[0]*b[1]+a[1]*b[0]);
	r[2] = montgomery_reduce(c[2]*(-147)+r[2]*zeta+a[0]*b[2]+a[1]*b[1]+a[2]*b[0]);
	r[3] = montgomery_reduce(c[3]*(-147)+a[0]*b[3]+a[1]*b[2]+a[2]*b[1]+a[3]*b[0]);
}
