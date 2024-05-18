

# setup python virtual environment
export ROOT_DIR=$(pwd)
virtualenv -p python3.10 venv/py3atsrtenv
source venv/py3atsrtenv/bin/activate
# download Llama3 model
huggingface-cli download meta-llama/Meta-Llama-3-70B-Instruct --local-dir Meta-Llama-3-70B-Instruct-HF
# clone TensorRT-LLM @ tag v0.9.0
git clone https://github.com/NVIDIA/TensorRT-LLM
cd TensorRT-LLM
git checkout -b tag-v0.9.0 v0.9.0
cd examples/llama
pip install tensorrt_llm=0.9.0 -U --pre --extra-index-url https://pypi.nvidia.com
pip install huggingface_hub pynvml mpi4py
pip install -r requirements.txt
cd $ROOT_DIR
# clone tensorrtllm_backend @ v0.9.0
git clone https://github.com/triton-inference-server/tensorrtllm_backend
cd tensorrtllm_backend
git checkout -b tag-v0.9.0 v0.9.0
cd $ROOT_DIR
# pull docker image
docker pull tritonserver:24.04-trtllm-python-py3


cd ${ROOT_DIR}
export HF_LLAMA_MODEL=${ROOT_DIR}/Meta-Llama-3-70B-Instruct-HF
export UNIFIED_CKPT_PATH=${ROOT_DIR}/TensorRT-LLM/examples/llama/tmp/trt_engines/4gpu/llama3-70b-instruct
export ENGINE_PATH=${ROOT_DIR}/TensorRT-LLM/examples/llama/tmp/trt_engines/compiled-model/4gpu/llama3-70b-instruct
CUDA_VISIBLE_DEVICES=0,2,3,4 python ${ROOT_DIR}/TensorRT-LLM/examples/llama/convert_checkpoint.py --model_dir ${HF_LLAMA_MODEL} --output_dir ${UNIFIED_CKPT_PATH} --dtype float16 --tp_size 4

CUDA_VISIBLE_DEVICES=0,2,3,4 trtllm-build --checkpoint_dir ${UNIFIED_CKPT_PATH} --remove_input_padding enable --max_input_len 8192 --gpt_attention_plugin float16 --context_fmha enable  --gemm_plugin float16  --output_dir ${ENGINE_PATH} --paged_kv_cache enable --max_batch_size 8
