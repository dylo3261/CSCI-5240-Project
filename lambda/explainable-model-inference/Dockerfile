FROM 029953330549.dkr.ecr.us-west-2.amazonaws.com/lambda-gdal-base:latest

RUN pip install --no-cache-dir scikit-learn==1.4.2

COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY terrain.py ${LAMBDA_TASK_ROOT}/
COPY weather.py ${LAMBDA_TASK_ROOT}/
COPY predict.py ${LAMBDA_TASK_ROOT}/
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

CMD ["lambda_function.lambda_handler"]
