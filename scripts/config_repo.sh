
# setup models repository
cd ${ROOT_DIR}/tensorrtllm_backend
mkdir Llama3-70B-Instruct
cp -rf all_models/inflight_batcher_llm/* Llama3-70B-Instruct/
# cp TensorRT engine images
cp -rf ${ROOT_DIR}/TensorRT-LLM/examples/llama/tmp/trt_engines/compiled-model/4gpu/llama3-70b-instruct/* Llama3-70B-Instruct/tensorrt_llm/1
# configure model repository parameters
export HOST_DIR=${ROOT_DIR}/tensorrtllm_backend/Llama3-70B-Instruct
export TARGET_DIRT=/all_models
export HF_MODEL=/Meta-Llama-3-70B-Instruct-HF

python3 ${ROO_DIR}/tensorrtllm_backend/tools/fill_template.py -i ${HOST_DIR}/preprocessing/config.pbtxt tokenizer_dir:${HF_MODEL},triton_max_batch_size:8,preprocessing_instance_count:1
python3 ${ROO_DIR}/tensorrtllm_backend/tools/fill_template.py -i ${HOST_DIR}/postprocessing/config.pbtxt tokenizer_dir:${HF_MODEL},triton_max_batch_size:8,postprocessing_instance_count:1
python3 ${ROO_DIR}/tensorrtllm_backend/tools/fill_template.py -i ${HOST_DIR}/tensorrt_llm_bls/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,bls_instance_count:1,accumulate_tokens:False
python3 ${ROO_DIR}/tensorrtllm_backend/tools/fill_template.py -i ${HOST_DIR}/ensemble/config.pbtxt triton_max_batch_size:8
python3 ${ROO_DIR}/tensorrtllm_backend/tools/fill_template.py -i ${HOST_DIR}/tensorrt_llm/config.pbtxt triton_max_batch_size:8,decoupled_mode:False,max_beam_width:1,engine_dir:${TARGET_DIR}/tensorrt_llm/1,exclude_input_in_output:True,enable_kv_cache_reuse:False,batching_strategy:inflight_fused_batching,max_queue_delay_microseconds:0
