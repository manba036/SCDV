FROM ubuntu:18.04

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  wget \
  curl \
  file \
  sudo \
  unzip \
  git \
  python3 \
  python3-pip \
  python3-dev \
  swig \
  mecab \
  libmecab-dev \
  mecab-ipadic-utf8

RUN wget -q https://github.com/neologd/mecab-ipadic-neologd/archive/master.zip -O mecab-ipadic-neologd.zip \
  && unzip mecab-ipadic-neologd.zip \
  && rm mecab-ipadic-neologd.zip \
  && sudo mecab-ipadic-neologd-master/bin/install-mecab-ipadic-neologd -n -a -y \
  && sudo sed -i -e "s|^dicdir.*$|dicdir = $(mecab-config --dicdir)/mecab-ipadic-neologd|" /etc/mecabrc \
  && rm -rf mecab-ipadic-neologd-master

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install jupyterlab ipywidgets
RUN jupyter nbextension enable --py widgetsnbextension

COPY ./requirements.txt /requirements.txt
RUN pip3 install -r /requirements.txt

RUN mkdir /work
WORKDIR /work

EXPOSE 8888

CMD ["jupyter", "lab", "--port=8888", "--ip=0.0.0.0", "--allow-root", "--no-browser"]