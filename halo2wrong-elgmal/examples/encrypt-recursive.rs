use std::path::PathBuf;
use ark_std::test_rng;
use halo2_snark_aggregator_circuit::sample_circuit::sample_circuit_setup;
use pairing_bn256::bn256::{Bn256, G1Affine};
use halo2wrong_elgmal::circuit::ElGamalTargetCircuit;
use halo2wrong_elgmal::{aggregation, ElGamalCircuit};

fn main() {
    let output_dir = PathBuf::from("./");
    let mut rnd = test_rng();
    sample_circuit_setup::<G1Affine, Bn256, ElGamalTargetCircuit>(output_dir.clone());

    let (_sk, pk) = ElGamalCircuit::<G1Affine>::keygen(&mut rnd).unwrap();

    aggregation::encrypt(output_dir.clone(), vec![1, 32], pk, &mut rnd);

    aggregation::verify_setup(output_dir.clone())

    // TODO verify_run
}
