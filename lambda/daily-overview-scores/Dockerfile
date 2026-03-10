FROM public.ecr.aws/lambda/python:3.11

RUN yum install -y \
    gcc10 gcc10-c++ make cmake3 automake libtool gzip tar \
    sqlite-devel curl-devel libtiff-devel libpng-devel libjpeg-devel \
    && yum clean all \
    && ln -sf /usr/bin/gcc10-gcc /usr/bin/gcc \
    && ln -sf /usr/bin/gcc10-g++ /usr/bin/g++ \
    && ln -sf /usr/bin/gcc10-gcc /usr/bin/cc \
    && ln -sf /usr/bin/gcc10-g++ /usr/bin/c++ \
    && ln -sf /usr/bin/gcc10-ar /usr/bin/ar \
    && ln -sf /usr/bin/gcc10-ar /usr/bin/gcc-ar \
    && ln -sf /usr/bin/gcc10-ranlib /usr/bin/ranlib \
    && ln -sf /usr/bin/gcc10-nm /usr/bin/nm \
    && ln -sf /usr/bin/cmake3 /usr/bin/cmake

RUN pip install --upgrade pip

RUN pip install --no-cache-dir "numpy==1.26.4" \
    && python -c "import numpy; print('numpy:', numpy.__version__)"

# Build SQLite 3.44
RUN curl -L https://www.sqlite.org/2023/sqlite-autoconf-3440200.tar.gz | tar xz \
    && cd sqlite-autoconf-3440200 \
    && ./configure --prefix=/usr/local \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf sqlite-autoconf-3440200 \
    && /sbin/ldconfig /usr/local/lib

# Build PROJ 7.2.1
ENV PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
RUN curl -L https://download.osgeo.org/proj/proj-7.2.1.tar.gz | tar xz \
    && cd proj-7.2.1 \
    && ./configure --prefix=/usr/local SQLITE3_CFLAGS="-I/usr/local/include" SQLITE3_LIBS="-L/usr/local/lib -lsqlite3" \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf proj-7.2.1 \
    && /sbin/ldconfig /usr/local/lib

# Build GDAL 3.4.3 (last version with top-level autoconf)
RUN curl -L https://github.com/OSGeo/gdal/releases/download/v3.4.3/gdal-3.4.3.tar.gz | tar xz \
    && cd gdal-3.4.3 \
    && ./configure --prefix=/usr/local --with-proj=/usr/local --with-libtiff --without-python --with-sqlite3=/usr/local \
    LDFLAGS="-L/usr/local/lib" CPPFLAGS="-I/usr/local/include" \
    && make -j$(nproc) \
    && make install \
    && cd .. && rm -rf gdal-3.4.3 \
    && /sbin/ldconfig /usr/local/lib \
    && echo "GDAL installed: $(/usr/local/bin/gdal-config --version)"

# Build rasterio against our GDAL
ENV GDAL_CONFIG=/usr/local/bin/gdal-config
RUN pip install --no-cache-dir setuptools cython \
    && pip install --no-cache-dir --no-build-isolation "rasterio>=1.3,<1.4" \
    && python -c "import rasterio; print('rasterio:', rasterio.__version__)" \
    && python -c "import numpy; print('numpy still:', numpy.__version__)"

# Install remaining ML deps
RUN pip install --no-cache-dir \
    "llvmlite==0.43.0" \
    "numba==0.60.0" \
    "scipy>=1.11,<1.14" \
    "pandas>=2.0.0,<2.3" \
    "scikit-learn>=1.3.0,<1.6" \
    "shap>=0.42.0,<0.50" \
    "joblib>=1.3.0" \
    "python-dotenv>=1.0.0" \
    "boto3>=1.28.0" \
    && python -c "import numpy; print('numpy final:', numpy.__version__)"


RUN python -c "import numpy, pandas, sklearn, shap, rasterio, joblib; print('ALL IMPORTS OK')"

# Copy grid Lambda files
COPY grid_lambda.py ${LAMBDA_TASK_ROOT}/
COPY terrain.py ${LAMBDA_TASK_ROOT}/
COPY weather.py ${LAMBDA_TASK_ROOT}/
COPY predict.py ${LAMBDA_TASK_ROOT}/
COPY utils/ ${LAMBDA_TASK_ROOT}/utils/

# Fix file permissions (Lambda needs read access)
RUN chmod -R 755 ${LAMBDA_TASK_ROOT}

# Grid Lambda handler
CMD ["grid_lambda.lambda_handler"]