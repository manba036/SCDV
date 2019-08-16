#!/bin/sh

if [ ! -e "./japanese-dataset/livedoor-news-corpus/dokujo-tsushin" ]; then
  wget https://www.rondhuit.com/download/ldcc-20140209.tar.gz
  tar xvfz ldcc-20140209.tar.gz -C ./japanese-dataset/livedoor-news-corpus
  rm -f ldcc-20140209.tar.gz
  cp -rf ./japanese-dataset/livedoor-news-corpus/text/* ./japanese-dataset/livedoor-news-corpus/
  rm -Rf ./japanese-dataset/livedoor-news-corpus/text
fi

docker build -t scdv ./docker_scdv
docker run --rm -it --name=scdv -p 8888:8888 -v `pwd`:/work/SCDV scdv