use ark_std::test_rng;
use halo2_proofs::arithmetic::Field;
use halo2_proofs::dev::MockProver;
use pasta_curves::arithmetic::CurveAffine;
use pasta_curves::group::Curve;
use pasta_curves::pallas;
use halo2_elgmal::{ElGamalConfig, ElGamalGadget};

fn main() {
    let mut rng = test_rng();

    let (_sk, pk) = ElGamalGadget::keygen(&mut rng).unwrap();

    let r = pallas::Scalar::random(&mut rng);
    let msg = pallas::Base::random(&mut rng);

    let circuit = ElGamalGadget::new(
        r,
        msg,
        pk,
    );

    let public_inputs = ElGamalGadget::get_instances(&circuit.resulted_ciphertext);

    MockProver::run(12, &circuit, public_inputs).unwrap();
}
