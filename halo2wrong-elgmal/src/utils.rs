use halo2_proofs::arithmetic::{BaseExt, FieldExt};
use halo2_proofs::transcript::{bn_to_field, field_to_bn};

pub fn base_to_scalar<B: BaseExt, S: FieldExt>(base: &B) -> S {
    let bn = field_to_bn(base);
    let modulus = field_to_bn(&-B::one()) + 1u64;
    let bn = bn % modulus;
    bn_to_field(&bn)
}
