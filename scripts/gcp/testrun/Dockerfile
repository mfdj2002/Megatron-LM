FROM nvcr.io/nvidia/pytorch:21.05-py3

ENV DEBIAN_FRONTEND noninteractive

RUN pip install --no-cache-dir transformers==4.30.0 \
    && pip install --no-cache-dir deepspeed==0.13.0 \
    && python -c "import deepspeed; print(deepspeed.__version__)" \
    && python -c "import transformers; print(transformers.__version__)"

COPY pretrain-gpt.sh /pretrain-gpt.sh
RUN chmod +x /pretrain-gpt.sh

RUN git clone https://github.com/scottchiefbaker/dool && \
	cd dool && \
	python install.py && \
	dool --version

ENTRYPOINT ["/pretrain-gpt.sh"]
