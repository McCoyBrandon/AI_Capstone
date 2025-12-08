#!/usr/bin/env bash

# Helper script to build and run the Docker image for the TabTransformer project.
# Usage:
#   ./run_docker.sh build        # build the image
#   ./run_docker.sh train        # train model -> runs/Test_run_1
#   ./run_docker.sh eval         # evaluate model from runs/Test_run_1/model.ckpt
#   ./run_docker.sh infer        # run inference/demo with Test_run_1 checkpoint
#   ./run_docker.sh all          # build + train + eval
#   ./run_docker.sh help         # show usage

set -e

IMAGE_NAME="tabtransformer-ai4i"
RUN_NAME="Test_run_1"

usage() {
  echo "Usage: $0 {build|train|eval|infer|all|help}"
  exit 1
}

if [ $# -lt 1 ]; then
  usage
fi

CMD="$1"
shift || true

case "$CMD" in
  build)
    echo "=== Building Docker image: ${IMAGE_NAME} ==="
    docker build -t "${IMAGE_NAME}" .
    ;;

  train)
    echo "=== Training model in Docker (run_name=${RUN_NAME}) ==="
    docker run --rm \
      -v "$(pwd)/runs:/app/runs" \
      -v "$(pwd)/src/data:/app/src/data" \
      "${IMAGE_NAME}" \
      python -m src.training.train --run_name "${RUN_NAME}"
    ;;

  eval)
    echo "=== Evaluating model in Docker (run_name=${RUN_NAME}) ==="
    docker run --rm \
      -v "$(pwd)/runs:/app/runs" \
      "${IMAGE_NAME}" \
      python -m src.training.eval \
        --ckpt "runs/${RUN_NAME}/model.ckpt" \
        --out  "runs/${RUN_NAME}/eval_metrics.json"
    ;;

  infer)
    echo "=== Running inference/demo in Docker (run_name=${RUN_NAME}) ==="
    docker run --rm \
      -v "$(pwd)/runs:/app/runs" \
      -v "$(pwd)/src/data:/app/src/data" \
      "${IMAGE_NAME}" \
      python -m src.deploy.infer \
        --ckpt "runs/${RUN_NAME}/model.ckpt" \
        --csv  "src/data/ai4i2020.csv" \
        --out  "runs/${RUN_NAME}/demo_metrics.json" \
        --failures-csv "runs/${RUN_NAME}/flagged_failures.csv"
    ;;

  all)
    echo "=== Building image, training, and evaluating (${RUN_NAME}) ==="
    "$0" build
    "$0" train
    "$0" eval
    ;;

  help|--help|-h)
    usage
    ;;

  *)
    echo "Unknown command: ${CMD}"
    usage
    ;;
esac
