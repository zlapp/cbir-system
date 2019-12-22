FROM alpine:3.7

RUN apk --update add python py-pip openssl ca-certificates
RUN apk --update add --virtual build-dependencies python-dev build-base wget \
  && apk del build-dependencies

ADD ./scripts/index_data.py index_data.py
ADD ./src src
ADD ./data/xsmall ./data

CMD ["python", "index_data.py", "-d", "./data"]