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
    BigInteger256 as BigInteger,
    bytes::{FromBytes, ToBytes},
    fields::{Field, FpParameters, LegendreSymbol, PrimeField, SquareRootField},
};
use curv::{FE,BigInt};
use curv::elliptic::curves::traits::ECScalar;

use curv::arithmetic::traits::Converter;
use curv::arithmetic::traits::Modulo;
use rand::{
    distributions::{Distribution, Standard},
    Rng,
};
use core::hash::{Hash, Hasher};
use std::ops::Shl;
use crate::io::Cursor;

#[derive(Derivative)]
#[derivative(
Clone(bound = ""),
Copy(bound = ""),
Debug(bound = ""),
PartialEq(bound = ""),
Eq(bound = "")
)]
pub struct FpCurv(pub FE);

impl Default for FpCurv {
    fn default() -> Self {
        FpCurv(FE::zero())
    }
}
// TODO: make sure this is not used
impl Hash for FpCurv {
    fn hash<H: Hasher>(&self, state: &mut H) {
        unimplemented!();
    }
}

impl UniformRand for FpCurv {
    fn rand<R: Rng + ?Sized>(rng: &mut R) -> Self{

        let mut bytes = [0u8;32];
        bytes  = rng.sample(Standard);
        let bn = BigInt::from(&bytes[..]);
        let fe : FE = ECScalar::from(&bn);

        FpCurv(fe)
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

fn zexe_biginteger_to_curv_bigint(src: &BigInteger) -> BigInt{
    //TODO: improve:
    let mut bytes: Vec<u8> = Vec::new();
    src.write( &mut bytes).unwrap();
    bytes.reverse();
    BigInt::from(&bytes[..])

}


fn curv_bigint_to_zexe_biginteger(src: &BigInt) -> BigInteger{
    let mut bytes = BigInt::to_vec(src);
    bytes.reverse();
    let cur = Cursor::new(bytes);
    let value = BigInteger::read(cur).unwrap();
    value

}

impl PrimeField for FpCurv {
    type Params = BLS12381Parameters;
    type BigInt = BigInteger;

    #[inline]
    fn from_repr(repr: <Self::Params as FpParameters>::BigInt) -> Self{
        let bn = zexe_biginteger_to_curv_bigint(&repr);
        FpCurv(ECScalar::from(&bn))
    }


    #[inline]
    fn into_repr(&self) -> Self::BigInt{
        let fe= self.0.clone();
        let bn = fe.to_big_int();
        curv_bigint_to_zexe_biginteger(&bn)
    }



    #[inline]
    fn multiplicative_generator() -> Self {
        let mg = zexe_biginteger_to_curv_bigint(&Self::Params::GENERATOR);
        let R2 = zexe_biginteger_to_curv_bigint(&Self::Params::R);
        let R2_inv = R2.invert(&FE::q()).unwrap();
        let mg_fix = BigInt::mod_mul(&mg, &R2_inv, &FE::q());
        FpCurv(ECScalar::from(&mg_fix))
    }

    #[inline]
    fn root_of_unity() -> Self {
        let rou_bn = zexe_biginteger_to_curv_bigint(&Self::Params::ROOT_OF_UNITY);
        let R2 = zexe_biginteger_to_curv_bigint(&Self::Params::R);
        let R2_inv = R2.invert(&FE::q()).unwrap();
        let rou_bn_fix = BigInt::mod_mul(&rou_bn, &R2_inv, &FE::q());
        FpCurv(ECScalar::from(&rou_bn_fix))

    }

}


impl Into<BigInteger> for FpCurv{
    fn into(self) -> BigInteger{
        Self::into_repr(&self)
    }

}

impl From<BigInteger> for FpCurv{
    fn from(other: BigInteger) -> Self{
        Self::from_repr(other)
    }
}

impl From<u128> for FpCurv{
    fn from(other: u128) -> Self{
        let upper = (other >> 64) as u64;
        let lower = ((other << 64) >> 64) as u64;
        let mut default_int = BigInteger::default();
        default_int.0[0] = lower;
        default_int.0[1] = upper;
        Self::from_repr(default_int)
    }
}

impl From<u8> for FpCurv{
    fn from(other: u8) -> Self{
        Self::from_repr(BigInteger::from(u64::from(other)))
    }
}
impl From<u16> for FpCurv{
    fn from(other: u16) -> Self{
        Self::from_repr(BigInteger::from(u64::from(other)))
    }
}
impl From<u32> for FpCurv{
    fn from(other: u32) -> Self{
        Self::from_repr(BigInteger::from(u64::from(other)))
    }
}
impl From<u64> for FpCurv{
    fn from(other: u64) -> Self{
        Self::from_repr(BigInteger::from(u64::from(other)))
    }
}

//TODO
impl ToBytes for FpCurv {
    #[inline]
    fn write<W: Write>(&self,mut writer: W) -> IoResult<()> {


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
        write!(f, "({})", self.0.to_big_int().to_str_radix(16))
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






pub struct BLS12381Parameters;

impl FpParameters for BLS12381Parameters {
    type BigInt = BigInteger;

    // MODULUS = 52435875175126190479447740508185965837690552500527637822603658699938581184513
    #[rustfmt::skip]
    const MODULUS: BigInteger = BigInteger([
        0xffffffff00000001,
        0x53bda402fffe5bfe,
        0x3339d80809a1d805,
        0x73eda753299d7d48,
    ]);

    const MODULUS_BITS: u32 = 255;

    const CAPACITY: u32 = Self::MODULUS_BITS - 1;

    const REPR_SHAVE_BITS: u32 = 1;

    #[rustfmt::skip]
    const R: BigInteger = BigInteger([
        0x1fffffffe,
        0x5884b7fa00034802,
        0x998c4fefecbc4ff5,
        0x1824b159acc5056f,
    ]);

    #[rustfmt::skip]
    const R2: BigInteger = BigInteger([
        0xc999e990f3f29c6d,
        0x2b6cedcb87925c23,
        0x5d314967254398f,
        0x748d9d99f59ff11,
    ]);

    const INV: u64 = 0xfffffffeffffffff;

    //
    #[rustfmt::skip]
    const GENERATOR: BigInteger = BigInteger([
        0xefffffff1,
        0x17e363d300189c0f,
        0xff9c57876f8457b0,
        0x351332208fc5a8c4,
    ]);

    const TWO_ADICITY: u32 = 32;

    #[rustfmt::skip]
    const ROOT_OF_UNITY: BigInteger = BigInteger([
        0xb9b58d8c5f0e466a,
        0x5b1b4c801819d7ec,
        0xaf53ae352a31e64,
        0x5bf3adda19e9b27b,
    ]);

    #[rustfmt::skip]
    const MODULUS_MINUS_ONE_DIV_TWO: BigInteger = BigInteger([
        0x7fffffff80000000,
        0xa9ded2017fff2dff,
        0x199cec0404d0ec02,
        0x39f6d3a994cebea4,
    ]);

    // T and T_MINUS_ONE_DIV_TWO, where MODULUS - 1 = 2^S * T

    // T = (MODULUS - 1) / 2^S =
    // 12208678567578594777604504606729831043093128246378069236549469339647
    #[rustfmt::skip]
    const T: BigInteger = BigInteger([
        0xfffe5bfeffffffff,
        0x9a1d80553bda402,
        0x299d7d483339d808,
        0x73eda753,
    ]);

    // (T - 1) / 2 =
    // 6104339283789297388802252303364915521546564123189034618274734669823
    #[rustfmt::skip]
    const T_MINUS_ONE_DIV_TWO: BigInteger = BigInteger([
        0x7fff2dff7fffffff,
        0x4d0ec02a9ded201,
        0x94cebea4199cec04,
        0x39f6d3a9,
    ]);
}


#[cfg(test)]
mod tests {

    use curv::{FE,BigInt};
    use curv::elliptic::curves::traits::ECScalar;
    use crate::BigInteger256 as BigInteger;
    use curv::arithmetic::traits::Converter;
    use crate::FpCurv;
    use crate::PrimeField;
    use crate::Field;
    use num_traits::identities::One;

    #[test]
    fn test_primefield_from_into_repr() {

        let fe : FE =ECScalar::new_random();
        let fp_curv = FpCurv(fe);
        let fp_to_repr = PrimeField::into_repr(&fp_curv);
        let fp_from_repr = PrimeField::from_repr(fp_to_repr);
        assert_eq!(fp_curv, fp_from_repr);
    }
    #[test]
    fn test_square() {
        let fe : FE =ECScalar::new_random();
        let mut fp_curv = FpCurv(fe);
        let fp_curv_1 = fp_curv.clone();
        fp_curv.square_in_place();
        let fp_curv_2 = fp_curv_1 * fp_curv_1;
        assert_eq!(fp_curv, fp_curv_2);
    }

    #[test]
    fn test_inv() {
        let size_as_bigint = BigInteger::from(2u64);
        let size_as_field_element = FpCurv::from_repr(size_as_bigint);
        let size_inv = size_as_field_element.inverse().unwrap();
        let size_inv_2 = size_inv.inverse().unwrap();
        assert_eq!(size_as_field_element, size_inv_2);
    }

    #[test]
    fn test_one() {
        let one_bigint = BigInteger::from(1u64);
        let fp1 = FpCurv::from_repr(one_bigint);
        let fp2 = FpCurv::one();
        assert_eq!(fp1, fp2);
    }

}


