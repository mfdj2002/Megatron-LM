FROM nvcr.io/nvidia/pytorch:23.04-py3

ENV DEBIAN_FRONTEND noninteractive

RUN pip install --no-cache-dir transformers==4.30.0 \
    && pip install --no-cache-dir deepspeed==0.13.0 \
    && python -c "import deepspeed; print(deepspeed.__version__)" \
    && python -c "import transformers; print(transformers.__version__)"

COPY ~/Megatron-LM/testrun/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]