use crate::circuit::{ElGamalCircuit, ElGamalTargetCircuit};
use crate::utils::base_to_scalar;
use ark_std::rand::{CryptoRng, RngCore};
use ark_std::{rand, test_rng};
use group::ff::{Field, PrimeField};
use group::Curve;
use halo2_snark_aggregator_circuit::fs::{write_verify_circuit_params, write_verify_circuit_vk};
use halo2_snark_aggregator_circuit::sample_circuit::{
    sample_circuit_random_run, sample_circuit_setup,
};
use halo2_snark_aggregator_circuit::verify_circuit::{MultiCircuitsSetup, Setup};
use pairing_bn256::arithmetic::CurveAffine;
use pairing_bn256::bn256::{Bn256, Fr, G1Affine, G1};
use rayon::prelude::*;
use std::path::PathBuf;

pub fn verify_setup(output: PathBuf) {
    let setup = [Setup::new::<ElGamalTargetCircuit>(&output)];

    let request = MultiCircuitsSetup::<_, _, 1> {
        setups: setup,
        coherent: vec![],
    };

    let (params, vk) = request.call(20);

    write_verify_circuit_params(&mut output.clone(), &params);
    write_verify_circuit_vk(&mut output.clone(), &vk);
}

pub fn encrypt<R: CryptoRng + RngCore>(
    setup_dir: PathBuf,
    plaintext: Vec<u8>,
    pk: G1,
    mut rnd: &mut R,
) {
    let r = Fr::random(&mut rnd);

    let msg = Fr::from_repr([1; 32]).unwrap();
    let circuit = ElGamalCircuit::<G1Affine>::new(pk.clone(), msg, r.clone()).unwrap();
    let c1_coordinates = circuit
        .resulted_ciphertext
        .0
        .to_affine()
        .coordinates()
        .map(|c| vec![c.x().clone(), c.y().clone()])
        .unwrap();
    let instances = vec![
        vec![base_to_scalar(&c1_coordinates[0])],
        vec![base_to_scalar(&c1_coordinates[1])],
        vec![circuit.resulted_ciphertext.1],
    ];

    sample_circuit_random_run::<G1Affine, Bn256, ElGamalTargetCircuit>(
        setup_dir.clone(),
        circuit,
        &instances
            .iter()
            .map(|instance| &instance[..])
            .collect::<Vec<_>>()[..],
        0,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_elgmal_setup() {
        let mut rnd = test_rng();
        sample_circuit_setup::<G1Affine, Bn256, ElGamalTargetCircuit>(PathBuf::from("./"));
    }

    #[test]
    fn test_elgmal_run() {
        let mut rnd = test_rng();
        let (_sk, pk) = ElGamalCircuit::<G1Affine>::keygen(&mut rnd).unwrap();

        encrypt(PathBuf::from("./"), vec![1, 2], pk, &mut rnd)
    }

    #[test]
    fn test_verify_setup() {
        let mut rng = test_rng();

        verify_setup(PathBuf::from("./"))
    }
}
