use crate::utils::base_to_scalar;
use ark_std::rand::{thread_rng, CryptoRng, RngCore};
use group::ff::Field;
use group::{Curve, Group};
use halo2_ecc_circuit_lib::chips::ecc_chip::{AssignedPoint, EccChipOps};
use halo2_ecc_circuit_lib::chips::integer_chip::{AssignedInteger, IntegerChipOps};
use halo2_ecc_circuit_lib::chips::native_ecc_chip::NativeEccChip;
use halo2_ecc_circuit_lib::five::base_gate::{FiveColumnBaseGate, FiveColumnBaseGateConfig};
use halo2_ecc_circuit_lib::five::integer_chip::FiveColumnIntegerChip;
use halo2_ecc_circuit_lib::five::range_gate::FiveColumnRangeGate;
use halo2_ecc_circuit_lib::gates::base_gate::{AssignedValue, Context};
use halo2_ecc_circuit_lib::gates::range_gate::RangeGateConfig;
use halo2_proofs::arithmetic::{CurveAffine, CurveExt, MultiMillerLoop};
use halo2_proofs::circuit::{Layouter, SimpleFloorPlanner};
use halo2_proofs::plonk;
use halo2_proofs::plonk::{Circuit, Column, ConstraintSystem, Error, Instance};
use halo2_snark_aggregator_api::arith::common::ArithCommonChip;
use halo2_snark_aggregator_api::hash::poseidon::PoseidonChip;
use halo2_snark_aggregator_circuit::chips::scalar_chip::ScalarChip;
use halo2_snark_aggregator_circuit::sample_circuit::TargetCircuit;
use std::ops::MulAssign;

#[derive(Debug, Clone)]
pub struct ElGamalConfig {
    base_gate_config: FiveColumnBaseGateConfig,
    range_gate_config: RangeGateConfig,
    c1_x: Column<Instance>,
    c1_y: Column<Instance>,
    c2: Column<Instance>,
}

const COMMON_RANGE_BITS: usize = 17usize;

#[derive(Default)]
pub struct ElGamalCircuit<C: CurveAffine> {
    rnd: C::ScalarExt,
    msg: C::ScalarExt,
    pk: C::CurveExt,
    pub resulted_ciphertext: (C::CurveExt, C::ScalarExt),
}

impl<C: CurveAffine> ElGamalCircuit<C> {
    pub fn new(pk: C::CurveExt, msg: C::ScalarExt, rnd: C::ScalarExt) -> anyhow::Result<Self> {
        let resulted_ciphertext = Self::encrypt(pk.clone(), msg.clone(), rnd.clone())?;
        Ok(Self {
            pk,
            msg,
            rnd,
            resulted_ciphertext,
        })
    }

    pub fn keygen<R: CryptoRng + RngCore>(
        mut rng: &mut R,
    ) -> anyhow::Result<(C::ScalarExt, C::CurveExt)> {
        // get a random element from the scalar field
        let secret_key = C::ScalarExt::random(&mut rng);

        // derive public_key = generator*secret_key
        let public_key = C::generator() * secret_key;

        Ok((secret_key, public_key))
    }

    pub fn encrypt(
        pk: C::CurveExt,
        msg: C::ScalarExt,
        r: C::ScalarExt,
    ) -> anyhow::Result<(C::CurveExt, C::ScalarExt)>
    where
        C::ScalarExt: halo2_proofs::arithmetic::FieldExt,
    {
        let mut c1 = C::CurveExt::generator();
        c1.mul_assign(r.clone());

        let (p_rx, p_ry, _) = (pk * r).jacobian_coordinates();
        let p_rx = base_to_scalar::<_, C::ScalarExt>(&p_rx);
        let p_ry = base_to_scalar::<_, C::ScalarExt>(&p_ry);

        let mut hasher = poseidon::Poseidon::<C::ScalarExt, 3, 2>::new(8usize, 33usize);
        hasher.update(&[p_rx.clone(), p_ry.clone()]);
        let dh = hasher.squeeze();
        let shared_key = dh;

        let c2 = msg + shared_key;

        return Ok((c1, c2));
    }

    pub fn get_instances(cipher: &(C::CurveExt, C::ScalarExt)) -> Vec<Vec<C::ScalarExt>> {
        let c1_coordinates = cipher
            .0
            .to_affine()
            .coordinates()
            .map(|c| vec![c.x().clone(), c.y().clone()])
            .unwrap();

        vec![
            vec![base_to_scalar(&c1_coordinates[1])],
            vec![base_to_scalar(&c1_coordinates[0])],
            vec![cipher.1],
        ]
    }

    pub(crate) fn verify_encryption<'a, 'b>(
        &self,
        ctx: &mut Context<'b, C::ScalarExt>,
        aff_chip: &ScalarChip<'a, 'b, C::ScalarExt>,
        ecc_chip: &NativeEccChip<'a, C>,
        msg: &AssignedValue<C::ScalarExt>,
    ) -> Result<(AssignedPoint<C, C::ScalarExt>, AssignedValue<C::ScalarExt>), plonk::Error> {
        // compute c1 = randomness*generator
        let mut c1 = C::CurveExt::generator();
        c1.mul_assign(self.rnd);

        let c1 = ecc_chip.assign_point(ctx, c1).unwrap();

        // compute s = randomness*pk
        let mut s = self.pk.clone();
        s.mul_assign(self.rnd);
        let si = ecc_chip.assign_point(ctx, s).unwrap();

        // compute dh = poseidon_hash(randomness*pk)
        // let mut hasher =
        //     PoseidonChip::<_, 3usize, 2usize>::new(ctx, aff_chip, 8usize, 33usize).unwrap();
        // let dh = {
        //     hasher.update(&[si.x.native.unwrap(), si.y.native.unwrap()]);
        //     hasher.squeeze(ctx, &aff_chip)
        // }
        // .unwrap();

        let dh = si.x.native.unwrap();

        // compute c2 = poseidon_hash(nk, rho) + msg.
        let c2 = aff_chip.add(ctx, &dh, msg).unwrap();

        Ok((c1, c2))
    }
}

impl<C: CurveAffine> Circuit<C::ScalarExt> for ElGamalCircuit<C>
where
    C::ScalarExt: halo2_proofs::arithmetic::Field,
{
    type Config = ElGamalConfig;
    type FloorPlanner = SimpleFloorPlanner;
    fn without_witnesses(&self) -> Self {
        Self::default()
    }
    //type Config = EccConfig;
    fn configure(meta: &mut ConstraintSystem<C::ScalarExt>) -> Self::Config {
        let base_gate_config = FiveColumnBaseGate::<C::ScalarExt>::configure(meta);
        let range_gate_config =
            FiveColumnRangeGate::<'_, C::Base, C::ScalarExt, COMMON_RANGE_BITS>::configure(
                meta,
                &base_gate_config,
            );
        let c1_x = meta.instance_column();
        meta.enable_equality(c1_x);
        let c1_y = meta.instance_column();
        meta.enable_equality(c1_y);
        let c2 = meta.instance_column();
        meta.enable_equality(c2);

        ElGamalConfig {
            base_gate_config,
            range_gate_config,
            c1_x,
            c1_y,
            c2,
        }
    }

    fn synthesize(
        &self,
        config: Self::Config,
        mut layouter: impl Layouter<C::ScalarExt>,
    ) -> Result<(), Error> {
        let base_gate = FiveColumnBaseGate::new(config.base_gate_config);
        let range_gate = FiveColumnRangeGate::<'_, C::Base, C::ScalarExt, COMMON_RANGE_BITS>::new(
            config.range_gate_config,
            &base_gate,
        );
        let integer_gate = FiveColumnIntegerChip::new(&range_gate);
        let scalar_chip = ScalarChip::new(&base_gate);
        let ecc_chip = NativeEccChip::new(&integer_gate);
        range_gate
            .init_table(&mut layouter, &integer_gate.helper.integer_modulus)
            .unwrap();

        let (c1, c2) = layouter.assign_region(
            || "elgmal_encryption",
            move |region| {
                let base_offset = 0usize;
                let mut ctx = Context::new(region, base_offset);
                let msg = scalar_chip.assign_var(&mut ctx, self.msg.clone()).unwrap();

                self.verify_encryption(&mut ctx, &scalar_chip, &ecc_chip, &msg)
            },
        )?;

        layouter
            .constrain_instance(c1.x.native.unwrap().cell, config.c1_x, 0)
            .and(layouter.constrain_instance(c1.y.native.unwrap().cell, config.c1_y, 0))
            .and(layouter.constrain_instance(c2.cell, config.c2, 0))
    }
}

pub struct ElGamalTargetCircuit;

impl<C: CurveAffine, E: MultiMillerLoop<G1Affine = C>> TargetCircuit<C, E>
    for ElGamalTargetCircuit
{
    const TARGET_CIRCUIT_K: u32 = 18;
    const PUBLIC_INPUT_SIZE: usize = 1;
    const N_PROOFS: usize = 2;
    const NAME: &'static str = "simple_example";
    const PARAMS_NAME: &'static str = "simple_example";

    type Circuit = ElGamalCircuit<C>;

    fn instance_builder() -> (Self::Circuit, Vec<Vec<C::ScalarExt>>) {
        let (sk, pk) = ElGamalCircuit::<C>::keygen(&mut thread_rng()).unwrap();

        let r = C::ScalarExt::random(&mut thread_rng());
        let msg = C::ScalarExt::random(&mut thread_rng());
        let resulted_ciphertext =
            ElGamalCircuit::<C>::encrypt(pk.clone(), msg.clone(), r.clone()).unwrap();

        let circuit = ElGamalCircuit::<C> {
            rnd: r,
            msg,
            pk,
            resulted_ciphertext,
        };

        let c1_coordinates = circuit
            .resulted_ciphertext
            .0
            .to_affine()
            .coordinates()
            .map(|c| vec![c.x().clone(), c.y().clone()])
            .unwrap();

        let instances = vec![
            vec![base_to_scalar(&c1_coordinates[1])],
            vec![base_to_scalar(&c1_coordinates[0])],
            vec![circuit.resulted_ciphertext.1],
        ];

        (circuit, instances)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::base_to_scalar;
    use anyhow::anyhow;
    use ark_std::test_rng;
    use group::Curve;
    use halo2_proofs::dev::MockProver;
    use pairing_bn256::bn256::{Fr, G1Affine};

    #[test]
    fn test_circuit_elgmal() {
        let mut rng = test_rng();

        let (_sk, pk) = ElGamalCircuit::<G1Affine>::keygen(&mut rng).unwrap();

        let r = Fr::random(&mut rng);
        let msg = Fr::random(&mut rng);
        let resulted_ciphertext =
            ElGamalCircuit::<G1Affine>::encrypt(pk.clone(), msg.clone(), r.clone()).unwrap();

        let circuit = ElGamalCircuit::<G1Affine> {
            rnd: r,
            msg,
            pk,
            resulted_ciphertext,
        };

        let c1_coordinates = circuit
            .resulted_ciphertext
            .0
            .to_affine()
            .coordinates()
            .map(|c| vec![c.x().clone(), c.y().clone()])
            .unwrap();
        let instance = vec![
            vec![base_to_scalar(&c1_coordinates[1])],
            vec![base_to_scalar(&c1_coordinates[0])],
            vec![circuit.resulted_ciphertext.1],
        ];
        let _prover = MockProver::run(18, &circuit, instance)
            .map_err(|e| anyhow!("error: {}", e))
            .unwrap();
    }
}
