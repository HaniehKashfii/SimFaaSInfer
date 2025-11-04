import argparse

from simfaasinfer.runtime_estimator.rf_estimator import RFEstimator
from simfaasinfer.utils.io import load_profile_artifacts


def main():
    parser = argparse.ArgumentParser(description="Train RFEstimator from profile artifacts.")
    parser.add_argument("--profiles_dir", default="/tmp/faasinfer_profiles", help="Directory with profile artifacts.")
    parser.add_argument("--out", default="/tmp/trained_estimator", help="Output directory to save estimator.")
    args = parser.parse_args()

    artifacts = load_profile_artifacts(args.profiles_dir)
    estimator = RFEstimator()
    model_spec = {"hidden_size": 4096, "num_layers": 32, "num_attention_heads": 32}
    hw_spec = {"mem_bw_gb_s": 2000, "tflops": 312, "nvlink_bw_gb_s": 600}

    estimator.fit(artifacts, model_spec=model_spec, hw_spec=hw_spec)
    estimator.save(args.out)
    print("Estimator trained and saved to", args.out)


if __name__ == "__main__":
    main()
