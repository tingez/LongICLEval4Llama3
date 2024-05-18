

cd ${ROOT_DIR}/tensorrtllm_backend
sudo docker run -it --rm --gpus '"device=0,2,3,4"' -p 9000:8000 -p 9001:8001 -p 9002:8002 --shm-size=4G -v ${ROO_DIR}/tensorrtllm_backend/Llama3-70B-Instruct:/all_models -v ${ROO_DIR}/tensorrtllm_backend/scripts:/opt/scripts -v ${ROOT_DIR}/Meta-Llama-3-70B-Instruct:/Meta-Llama-3-70B-Instruct-HF nvcr.io/nvidia/tritonserver:24.04-trtllm-python-py3
# enter tritonserver:24.04-trtllm-python-py3 docker
python /opt/scripts/launch_triton_server.py --world_size 4 --model_repo=/all_models
