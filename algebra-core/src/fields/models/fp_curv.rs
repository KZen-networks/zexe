use num_traits::{One, Zero};
use core::{
    cmp::{Ord, Ordering, PartialOrd},
    fmt::{Display, Formatter, Result as FmtResult},
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
    str::FromStr,
};
use crate::{
    io::{Read, Result as IoResult, Write},
    CanonicalDeserialize, CanonicalDeserializeWithFlags, CanonicalSerialize,
    CanonicalSerializeWithFlags, ConstantSerializedSize, EmptyFlags, Flags, SerializationError,
    UniformRand,
};
use crate::{
 //   biginteger::{arithmetic as fa, BigInteger as _BigInteger, BigInteger256 as BigInteger},
    BigInteger256 as BigInteger,
    bytes::{FromBytes, ToBytes},
    fields::{Field, FpParameters, LegendreSymbol, PrimeField, SquareRootField},
};
use curv::{FE,BigInt};
use curv::elliptic::curves::traits::ECScalar;

use curv::arithmetic::traits::Converter;

use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use core::hash::{Hash, Hasher};


#[derive(Derivative)]
#[derivative(
Clone(bound = ""),
Copy(bound = ""),
Debug(bound = ""),
PartialEq(bound = ""),
Eq(bound = "")
)]
pub struct FpCurv(pub FE);
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
impl Default for FpCurv {
    fn default() -> Self {
        FpCurv(FE::zero())
    }
}
// TODO: make sure this is not used
impl Hash for FpCurv {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unimplemented!();
 //       let zero :u32 = 0;
 //       zero.hash(state);
    }
}

impl UniformRand for FpCurv {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self{
        FpCurv(ECScalar::new_random())
    }
}

impl CanonicalDeserialize for FpCurv {
    #[inline]
    fn deserialize<R: Read>(reader: &mut R) -> Result<Self, SerializationError> {
        let mut bytes = [0u8; 32];
        reader.read_exact(&mut bytes)?;
        let v = BigInt::from(&bytes[..]);
        let fe: FE =ECScalar::from(&v);
        Ok(FpCurv(fe))
    }
}


impl ConstantSerializedSize for FpCurv {
    const SERIALIZED_SIZE: usize = 32;
    const UNCOMPRESSED_SIZE: usize = Self::SERIALIZED_SIZE;
}



impl CanonicalDeserializeWithFlags for FpCurv {
    /// Reads `Self` and `Flags` from `reader`.
    /// Returns empty flags by default.
    fn deserialize_with_flags<R: Read, F: Flags>(
        reader: &mut R,
    ) -> Result<(Self, F), SerializationError>{
        let mut bytes = [0u8; 32];
        reader.read_exact(&mut bytes)?;
        let v = BigInt::from(&bytes[..]);
        let fe: FE =ECScalar::from(&v);
        // TODO: make sure not used
        let flags = F::from_u8_remove_flags(&mut bytes[0]);
        Ok((FpCurv(fe),flags))
    }
}

/// Serializer in little endian format allowing to encode flags.
impl CanonicalSerializeWithFlags for FpCurv {
    /// Serializes `self` and `flags` into `writer`.
    fn serialize_with_flags<W: Write, F: Flags>(
        &self,
        writer: &mut W,
        flags: F,
    ) -> Result<(), SerializationError>{
        let bn = self.0.to_big_int();
        let bytes = BigInt::to_vec(&bn);
        Ok(writer.write_all(&bytes[..])?)
    }
}

impl CanonicalSerialize for FpCurv {
    #[inline]
    fn serialize<W: Write>(&self, writer: &mut W) -> Result<(), SerializationError> {
        let bn = self.0.to_big_int();
        let bytes = BigInt::to_vec(&bn);
        Ok(writer.write_all(&bytes[..])?)
    }

    #[inline]
    fn serialized_size(&self) -> usize {
        Self::SERIALIZED_SIZE
    }
}
/*
    /// Serializes `self` into `writer` without compression.
    #[inline]
    fn serialize_uncompressed<W: Write>(&self, writer: &mut W) -> Result<(), SerializationError> {
        self.serialize(writer)
    }
    #[inline]
    fn uncompressed_size(&self) -> usize {
        self.serialized_size()
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


const MODULUS: BigInteger = BigInteger([
    0xFFFFFFFFFFFFFFFF,
    0xFFFFFFFFFFFFFFFE,
    0xBAAEDCE6AF48A03B,
    0xBFD25E8CD0364141,
]);

impl Field for FpCurv {

    // TODO: ask for change upstream
    #[inline]
    fn characteristic<'a>() -> &'a [u64] {
        let q_bn = FE::q();
        let mut u64_limbs: Vec<u64> =  Vec::new();
        // assuming size of field element is between 192-256 bits
        let two = BigInt::from(2);
        let mut arr = [0u8;8];
        let filter = two.pow(64) - BigInt::one();
        u64_limbs = (0..4).map(|i|{
            let limb_bn : BigInt =  ( q_bn.clone() >> ( 64 * i)) & filter.clone();
            let mut bytes = BigInt::to_vec(&limb_bn);
            bytes.reverse();
            arr.copy_from_slice(&bytes[..]);
            u64::from_le_bytes(arr.clone())
        }).collect::<Vec<u64>>();
        MODULUS.as_ref()
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
    fn write<W: Write>(&self,mut writer: W) -> IoResult<()> {
       // let bn = self.0.to_big_int();
       // let bytes = BigInt::to_vec(&bn);
       // write!(writer, "{}", &bytes[..])

        let bn = self.0.to_big_int();
        let bytes = BigInt::to_vec(&bn);
        Ok(writer.write_all(&bytes[..])?)

    }
}

impl FromBytes for FpCurv {
    #[inline]
    fn read<R: Read>(mut reader: R) -> IoResult<Self> {
        let mut bytes = [0u8; 32];
        reader.read_exact(&mut bytes)?;
        let v = BigInt::from(&bytes[..]);
        let fe: FE = ECScalar::from(&v);
        Ok(FpCurv(fe))
    }
}


/// `Fp` elements are ordered lexicographically.
impl Ord for FpCurv {
    #[inline(always)]
    fn cmp(&self, other: &Self) -> Ordering {
        let a = self.0.to_big_int();
        let b = other.0.to_big_int();
        if a < b {
            return core::cmp::Ordering::Less;
        } else if a > b {
            return core::cmp::Ordering::Greater;
        }
    core::cmp::Ordering::Equal
    }
}
impl PartialOrd for FpCurv {
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
        write!(f, "({})", self.0.get_element())
    }
}

impl Neg for FpCurv {
    type Output = Self;
    #[inline]
    #[must_use]
    fn neg(self) -> Self {
        let neg_bn = FE::q() - &self.0.to_big_int();
        let neg_fe :FE = ECScalar::from(&neg_bn);
        FpCurv(neg_fe)
    }
}

impl Add<FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn add(self, other: Self) -> Self {

        let mut result = self;
        result.0 = result.0 + other.0;
        result
    }
}

impl<'a> Add<&'a FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn add(self, other: &Self) -> Self {
        let mut result = self;
        result.0 = result.0 + other.0;
        result
    }
}


impl Sub<FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn sub(self, other: Self) -> Self {
        let mut result = self;
        result.0 = result.0.sub( &other.0.get_element());
        result
    }
}


impl<'a> Sub<&'a FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn sub(self, other: &Self) -> Self {
        let mut result = self;
        result.0 = result.0.sub( &other.0.get_element());
        result
    }
}

impl Mul<FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn mul(self, other: Self) -> Self {
        let mut result = self;
        result.0 = result.0 * other.0;
        result
    }
}

impl<'a> Mul<&'a FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn mul(self, other: &Self) -> Self {
        let mut result = self;
        result.0 = result.0 * other.0;
        result
    }
}

impl Div<FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn div(self, other: Self) -> Self {
        let mut result = self;
        result.0 = result.0 * &other.0.invert();
        result
    }
}

impl<'a> Div<&'a FpCurv> for FpCurv {
    type Output = Self;

    #[inline]
    fn div(self, other: &Self) -> Self {
        let mut result = self;
        result.0 = result.0 * other.0.invert();
        result
    }
}

//impl_addassign_from_ref!(Fp256, Fp256Parameters);

impl AddAssign<Self> for FpCurv {
    #[inline]
    fn add_assign(&mut self, other: Self) {
        self.0 = self.0 + &other.0;
    }
}

impl<'a> AddAssign<&'a Self> for FpCurv {
    #[inline]
    fn add_assign(&mut self, other: &Self) {
        self.0 = self.0 + other.0;
    }
}

impl SubAssign<Self> for FpCurv {
    #[inline]
    fn sub_assign(&mut self, other: Self) {
        self.0 = self.0.sub(&other.0.get_element());
    }
}

impl<'a> SubAssign<&'a Self> for FpCurv {
    #[inline]
    fn sub_assign(&mut self, other: &Self) {
        self.0 = self.0.sub(&other.0.get_element());
    }
}

impl MulAssign<Self> for FpCurv {
    #[inline]
    fn mul_assign(&mut self, other: Self) {
        self.0 = self.0 * &other.0;
    }
}

impl<'a> MulAssign<&'a Self> for FpCurv {
    #[inline]
    fn mul_assign(&mut self, other: &Self) {
        self.0 = self.0 * other.0;
    }
}

impl DivAssign<Self> for FpCurv {
    #[inline]
    fn div_assign(&mut self, other: Self) {
        self.0 = self.0 * &other.0.invert();
    }
}

impl<'a> DivAssign<&'a Self> for FpCurv {
    #[inline]
    fn div_assign(&mut self, other: &Self) {
        self.0 = self.0 * other.0.invert();
    }
}




impl core::iter::Sum<Self> for FpCurv {
fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
iter.fold(Self::zero(), |acc,x |FpCurv(x.0 + acc.0))
}
}

#[allow(unused_qualifications)]
impl<'a> core::iter::Sum<&'a Self> for FpCurv {
fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
iter.fold(Self::zero(), |acc,x |FpCurv(x.0 + acc.0))

}
}

impl core::iter::Product<Self> for FpCurv {
fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
    iter.fold(Self::one(), |acc,x | FpCurv(acc.0 * x.0))
}
}

impl<'a> core::iter::Product<&'a Self> for FpCurv {
fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
iter.fold(Self::one(), |acc,x | FpCurv(acc.0 * x.0))
}
}