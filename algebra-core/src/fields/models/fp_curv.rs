use num_traits::{One, Zero};
use core::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Display, Formatter, Result as FmtResult},
    io::{Read, Result as IoResult, Write},
    marker::PhantomData,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};

use crate::{
    biginteger::{arithmetic as fa, BigInteger as _BigInteger, BigInteger256 as BigInteger},
    bytes::{FromBytes, ToBytes},
    fields::{Field, FpParameters, LegendreSymbol, PrimeField, SquareRootField},
};
use curv::{FE,BigInt};
use curv::elliptic::curves::traits::ECScalar;
use core::fmt::Write;
use crate::io::Read;
use serde::ser::Serialize;
use curv::arithmetic::traits::Converter;


#[derive(Derivative)]
#[derivative(
Default(bound = ""),
Hash(bound = ""),
Clone(bound = ""),
Copy(bound = ""),
Debug(bound = ""),
PartialEq(bound = ""),
Eq(bound = "")
)]
pub struct FpCurv(FE);
/*
impl<P> Fp256<P> {
    #[inline]
    pub const fn new(element: BigInteger) -> Self {
        Self(element, PhantomData)
    }
}

*/

/*
impl<P: Fp256Parameters> Fp256<P> {
    #[inline]
    fn is_valid(&self) -> bool {
        self.0 < P::MODULUS
    }

    #[inline]
    fn reduce(&mut self) {
        if !self.is_valid() {
            self.0.sub_noborrow(&P::MODULUS);
        }
    }

    #[inline]
    fn mont_reduce(
        &mut self,
        r0: u64,
        mut r1: u64,
        mut r2: u64,
        mut r3: u64,
        mut r4: u64,
        mut r5: u64,
        mut r6: u64,
        mut r7: u64,
    ) {
        // The Montgomery reduction here is based on Algorithm 14.32 in
        // Handbook of Applied Cryptography
        // <http://cacr.uwaterloo.ca/hac/about/chap14.pdf>.

        let k = r0.wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r0, k, P::MODULUS.0[0], &mut carry);
        r1 = fa::mac_with_carry(r1, k, P::MODULUS.0[1], &mut carry);
        r2 = fa::mac_with_carry(r2, k, P::MODULUS.0[2], &mut carry);
        r3 = fa::mac_with_carry(r3, k, P::MODULUS.0[3], &mut carry);
        r4 = fa::adc(r4, 0, &mut carry);
        let carry2 = carry;
        let k = r1.wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r1, k, P::MODULUS.0[0], &mut carry);
        r2 = fa::mac_with_carry(r2, k, P::MODULUS.0[1], &mut carry);
        r3 = fa::mac_with_carry(r3, k, P::MODULUS.0[2], &mut carry);
        r4 = fa::mac_with_carry(r4, k, P::MODULUS.0[3], &mut carry);
        r5 = fa::adc(r5, carry2, &mut carry);
        let carry2 = carry;
        let k = r2.wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r2, k, P::MODULUS.0[0], &mut carry);
        r3 = fa::mac_with_carry(r3, k, P::MODULUS.0[1], &mut carry);
        r4 = fa::mac_with_carry(r4, k, P::MODULUS.0[2], &mut carry);
        r5 = fa::mac_with_carry(r5, k, P::MODULUS.0[3], &mut carry);
        r6 = fa::adc(r6, carry2, &mut carry);
        let carry2 = carry;
        let k = r3.wrapping_mul(P::INV);
        let mut carry = 0;
        fa::mac_with_carry(r3, k, P::MODULUS.0[0], &mut carry);
        r4 = fa::mac_with_carry(r4, k, P::MODULUS.0[1], &mut carry);
        r5 = fa::mac_with_carry(r5, k, P::MODULUS.0[2], &mut carry);
        r6 = fa::mac_with_carry(r6, k, P::MODULUS.0[3], &mut carry);
        r7 = fa::adc(r7, carry2, &mut carry);
        (self.0).0[0] = r4;
        (self.0).0[1] = r5;
        (self.0).0[2] = r6;
        (self.0).0[3] = r7;
        self.reduce();
    }
}
*/

impl Zero for FpCurv {
    #[inline]
    fn zero() -> Self {
        FpCurv(FE::zero())
    }
    #[inline]
    fn is_zero(&self) -> bool {
        self.0.to_big_int() == BigInt::zero()
    }
}

impl One for FpCurv{
    #[inline]
    fn one() -> Self {
        let one = BigInt::one();
        FpCurv(ECScalar::from(&one))
    }

    #[inline]
    fn is_one(&self) -> bool {
        self.0.to_big_int() == BigInt::one()
    }
}

impl Field for FpCurv {

    #[inline]
    fn characteristic<'a>() -> &'a [u64] {
        let mut q_bn = FE::q();
        // assuming size of field element is between 192-256 bits
        let two = BigInt::from(2);
        let mut filter = two.pow(64) - BigInt::one();
        let u64_limbs = (0..4).map(|i|{
            ( q_bn >> ( 64 * i)) && filter
        }).collect::<Vec<u64>>();
        &u64_limbs[..]
    }


    #[inline]
    fn double(&self) -> Self {
        FpCurv(self.0 + &self.0)

    }

    #[inline]
    fn double_in_place(&mut self) -> &mut Self {
        let two = FpCurv::one().double();
        self.0 = self.0 * two.0;
        self
    }


    #[inline]
    fn from_random_bytes_with_flags(bytes: &[u8]) -> Option<(Self, u8)>{
        let bn = BigInt::from(bytes);
        let fe :FE = ECScalar::from(&bn);
        // TODO: what is this byte used for??
        Some((FpCurv(fe), 0))
    }


    #[inline]
    fn square(&self) -> Self {
        FpCurv(self.0 * &self.0)
    }

    #[inline]
    fn square_in_place(&mut self) -> &mut Self {
        self.0 = self.0 * &self.0;
        self
    }

    #[inline]
    fn inverse(&self) -> Option<Self> {
        Some(FpCurv(self.0.invert()))
    }

    fn inverse_in_place(&mut self) -> Option<&mut Self> {
        self.0 = self.0.invert();
        Some(self)
    }

    #[inline]
    fn frobenius_map(&mut self, _: usize) {
        // No-op: No effect in a prime field.
    }


}
/*
impl<P: Fp256Parameters> PrimeField for Fp256<P> {
    type Params = P;
    type BigInt = BigInteger;

    #[inline]
    fn from_repr(r: BigInteger) -> Self {
        let mut r = Fp256(r, PhantomData);
        if r.is_valid() {
            r.mul_assign(&Fp256(P::R2, PhantomData));
            r
        } else {
            Self::zero()
        }
    }

    #[inline]
    fn into_repr(&self) -> BigInteger {
        let mut r = *self;
        r.mont_reduce(
            (self.0).0[0],
            (self.0).0[1],
            (self.0).0[2],
            (self.0).0[3],
            0,
            0,
            0,
            0,
        );
        r.0
    }



    #[inline]
    fn multiplicative_generator() -> Self {
        Fp256::<P>(P::GENERATOR, PhantomData)
    }

    #[inline]
    fn root_of_unity() -> Self {
        Fp256::<P>(P::ROOT_OF_UNITY, PhantomData)
    }
}

impl<P: Fp256Parameters> SquareRootField for Fp256<P> {
    #[inline]
    fn legendre(&self) -> LegendreSymbol {
        use crate::fields::LegendreSymbol::*;

        // s = self^((MODULUS - 1) // 2)
        let s = self.pow(P::MODULUS_MINUS_ONE_DIV_TWO);
        if s.is_zero() {
            Zero
        } else if s.is_one() {
            QuadraticResidue
        } else {
            QuadraticNonResidue
        }
    }

    // Only works for p = 1 (mod 16).
    #[inline]
    fn sqrt(&self) -> Option<Self> {
        sqrt_impl!(Self, P, self)
    }

    fn sqrt_in_place(&mut self) -> Option<&mut Self> {
        if let Some(sqrt) = self.sqrt() {
            *self = sqrt;
            Some(self)
        } else {
            None
        }
    }
}

*/
/*
impl_prime_field_from_int!(Fp256, u128, Fp256Parameters);
impl_prime_field_from_int!(Fp256, u64, Fp256Parameters);
impl_prime_field_from_int!(Fp256, u32, Fp256Parameters);
impl_prime_field_from_int!(Fp256, u16, Fp256Parameters);
impl_prime_field_from_int!(Fp256, u8, Fp256Parameters);

impl_prime_field_standard_sample!(Fp256, Fp256Parameters);
*/
//TODO
impl ToBytes for FpCurv {
    #[inline]
    fn write<W: Write>(&self, writer: W) -> IoResult<()> {
        write!(self.0.to_big_int().to_hex());
        Ok(())
    }
}

impl FromBytes for FpCurv {
    #[inline]
    fn read<R: Read>(reader: R) -> IoResult<Self> {

        let v = BigInt::from_str_radix(reader, 16).expect("Failed in serde");
        Ok(FpCurv(ECScalar::from(&v))   ) }
}

//TODO
/// `Fp` elements are ordered lexicographically.
impl<P: Fp256Parameters> Ord for Fp256<P> {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        self.into_repr().cmp(&other.into_repr())
    }
}
//TODO
impl<P: Fp256Parameters> PartialOrd for Fp256<P> {
    #[inline(always)]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl FromStr for FpCurv {
    type Err = ();

    /// Interpret a string of numbers as a (congruent) prime field element.
    /// Does not accept unnecessary leading zeroes or a blank string.
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let v = BigInt::from_str_radix(s, 10).expect("Failed in serde");
        Ok(FpCurv(ECScalar::from(&v))   ) }

}

impl Display for FpCurv {
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "({})", self.0)
    }
}

impl Neg for FpCurv {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        let neg_bn = FE::q() - &self.0.to_big_int();
        let neg_fe :FE = ECScalar::from(neg_bn);
        FpCurv(neg_fe)
    }
}

impl Add<FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {

        let mut result = self;
        result.0 = result.0 + result.0;
        result
    }
}

impl<'a> Add<&'a FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self;
        result.0 = result.0 + result.0;
        result
    }
}


//TODO
impl<'a, P: Fp256Parameters> Sub<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        let mut result = self;
        result.sub_assign(other);
        result
    }
}

impl<P: Fp256Parameters> Mul<Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut result = self;
        result.mul_assign(&other);
        result
    }
}

impl<'a, P: Fp256Parameters> Mul<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(other);
        result
    }
}

impl<'a, P: Fp256Parameters> Div<&'a Fp256<P>> for Fp256<P> {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        let mut result = self;
        result.mul_assign(&other.inverse().unwrap());
        result
    }
}

impl_addassign_from_ref!(Fp256, Fp256Parameters);
impl<'a, P: Fp256Parameters> AddAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        // This cannot exceed the backing capacity.
        self.0.add_nocarry(&other.0);
        // However, it may need to be reduced

        self.reduce();
    }
}

impl<'a, P: Fp256Parameters> SubAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        // If `other` is larger than `self`, add the modulus to self first.
        if other.0 > self.0 {
            self.0.add_nocarry(&P::MODULUS);
        }

        self.0.sub_noborrow(&other.0);
    }
}

impl_mulassign_from_ref!(Fp256, Fp256Parameters);
impl<'a, P: Fp256Parameters> MulAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        let mut carry = 0;
        let r0 = fa::mac_with_carry(0, (self.0).0[0], (other.0).0[0], &mut carry);
        let r1 = fa::mac_with_carry(0, (self.0).0[0], (other.0).0[1], &mut carry);
        let r2 = fa::mac_with_carry(0, (self.0).0[0], (other.0).0[2], &mut carry);
        let r3 = fa::mac_with_carry(0, (self.0).0[0], (other.0).0[3], &mut carry);
        let r4 = carry;
        let mut carry = 0;
        let r1 = fa::mac_with_carry(r1, (self.0).0[1], (other.0).0[0], &mut carry);
        let r2 = fa::mac_with_carry(r2, (self.0).0[1], (other.0).0[1], &mut carry);
        let r3 = fa::mac_with_carry(r3, (self.0).0[1], (other.0).0[2], &mut carry);
        let r4 = fa::mac_with_carry(r4, (self.0).0[1], (other.0).0[3], &mut carry);
        let r5 = carry;
        let mut carry = 0;
        let r2 = fa::mac_with_carry(r2, (self.0).0[2], (other.0).0[0], &mut carry);
        let r3 = fa::mac_with_carry(r3, (self.0).0[2], (other.0).0[1], &mut carry);
        let r4 = fa::mac_with_carry(r4, (self.0).0[2], (other.0).0[2], &mut carry);
        let r5 = fa::mac_with_carry(r5, (self.0).0[2], (other.0).0[3], &mut carry);
        let r6 = carry;
        let mut carry = 0;
        let r3 = fa::mac_with_carry(r3, (self.0).0[3], (other.0).0[0], &mut carry);
        let r4 = fa::mac_with_carry(r4, (self.0).0[3], (other.0).0[1], &mut carry);
        let r5 = fa::mac_with_carry(r5, (self.0).0[3], (other.0).0[2], &mut carry);
        let r6 = fa::mac_with_carry(r6, (self.0).0[3], (other.0).0[3], &mut carry);
        let r7 = carry;
        self.mont_reduce(r0, r1, r2, r3, r4, r5, r6, r7);
    }
}

impl<'a, P: Fp256Parameters> DivAssign<&'a Self> for Fp256<P> {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.mul_assign(&other.inverse().unwrap());
    }
}
