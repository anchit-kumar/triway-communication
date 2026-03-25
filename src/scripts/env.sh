source /ssd/SciencePrj25/venv/tf_train/bin/activate
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra:/usr/lib/aarch64-linux-gnu:/lib/aarch64-linux-gnu:/usr/local/cuda/targets/aarch64-linux/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
export PATH=/usr/local/cuda/bin:${PATH}
export MODELS_DIR=/ssd/SciencePrj25/src/models
export HANDPOSE_MODEL=${MODELS_DIR}/hand_landmarker.task
