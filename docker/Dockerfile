FROM python:3.6

# Box2d requires SWIG >= 3
# SWIG from YUM is version 2.0.1.1
# https://github.com/openai/gym/blob/box2dtests/.travis.yml#L20-L22
RUN wget https://downloads.sourceforge.net/project/swig/swig/swig-3.0.12/swig-3.0.12.tar.gz \
    && tar xf swig-3.0.12.tar.gz \
    && rm -f swig-3.0.12.tar.gz \
    && cd swig-3.0.12 \
    && ./configure \
    && make \
    && make install \
    && cd .. \
    && rm -rf swig-3.0.12

# CMake is required for Box2d
RUN wget https://cmake.org/files/v3.11/cmake-3.11.4-Linux-x86_64.sh \
    && chmod +x cmake-3.11.4-Linux-x86_64.sh \
    && ./cmake-3.11.4-Linux-x86_64.sh --skip-license \
    && rm -f cmake-3.11.4-Linux-x86_64.sh

RUN apt-get update && apt-get install -y python-opengl && apt-get clean

COPY requirements.txt /
RUN pip install -r requirements.txt

ENV HOME=/app
ENV PYTHONPATH=$HOME

COPY /deeprl $HOME/deeprl
COPY /examples $HOME/examples
COPY setup.py $HOME

ENTRYPOINT ["python"]

