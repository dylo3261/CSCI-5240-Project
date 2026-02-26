FROM public.ecr.aws/lambda/python:3.13

COPY requirements.txt .
RUN pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

COPY lambda_function.py ${LAMBDA_TASK_ROOT}
COPY update_caic_data.py ${LAMBDA_TASK_ROOT}
COPY update_weather_data.py ${LAMBDA_TASK_ROOT}
COPY utils ${LAMBDA_TASK_ROOT}/utils

CMD ["lambda_function.lambda_handler"]