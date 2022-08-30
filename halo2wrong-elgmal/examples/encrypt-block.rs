use ark_std::test_rng;
use group::ff::Field;
use halo2_proofs::dev::MockProver;
use pairing_bn256::bn256::{Fr, G1Affine};
use halo2wrong_elgmal::circuit::ElGamalCircuit;

type Circuit = ElGamalCircuit::<G1Affine>;

fn main() {
    let mut rng = test_rng();

    let (_sk, pk) = Circuit::keygen(&mut rng).unwrap();

    let r = Fr::random(&mut rng);
    let msg = Fr::random(&mut rng);

    let circuit = Circuit::new(
        pk,
        msg,
        r
    ).unwrap();

    let public_inputs = Circuit::get_instances(&circuit.resulted_ciphertext);

    MockProver::run(18, &circuit, public_inputs)
        .unwrap();
}
