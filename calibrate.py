import argparse

from simfaasinfer.calibration.calibrator import calibrate
from simfaasinfer.runtime_estimator.rf_estimator import RFEstimator
from simfaasinfer.utils.io import load_telemetry_samples


def main():
    parser = argparse.ArgumentParser(description="Calibrate a trained RFEstimator using telemetry samples.")
    parser.add_argument("--estimator_dir", default="/tmp/trained_estimator", help="Directory of trained estimator.")
    parser.add_argument("--telemetry", default="/tmp/telemetry.json", help="Telemetry samples JSON file.")
    parser.add_argument("--out", default="/tmp/calibrated_estimator", help="Output directory for calibrated model.")
    args = parser.parse_args()

    estimator = RFEstimator()
    estimator.load(args.estimator_dir)
    telemetry = load_telemetry_samples(args.telemetry)

    calibrated = calibrate(estimator, telemetry)
    calibrated.base_estimator.save(args.out)
    print("Calibrated estimator saved to", args.out)


if __name__ == "__main__":
    main()
