## Stage 1: Build the TF Binary H5 Converter
FROM python:3.7.5-buster as python-builder

WORKDIR /usr/src/app

COPY evaluator-tensorflow/h5_converter /usr/src/app

RUN make

## Stage 2: Build the full project
FROM maven:3.6.1-jdk-11-slim AS base

RUN apt-get update && \
    apt-get install make libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY --from=python-builder /usr/src/app/dist/h5_converter /usr/src/bin/h5_converter
COPY . /usr/src/app/

RUN make build MAVEN_PROFILE=full

## Stage 3 : create the docker final image
FROM centos:centos8

USER root

RUN mkdir -p /deployments

# JAVA_APP_DIR is used by run-java.sh for finding the binaries
ENV JAVA_APP_DIR=/deployments \
    JAVA_MAJOR_VERSION=11

# /dev/urandom is used as random source, which is prefectly safe
# according to http://www.2uo.de/myths-about-urandom/
RUN yum install -y \
       java-11-openjdk.x86_64 \
       java-11-openjdk-devel.x86_64 \
    && echo "securerandom.source=file:/dev/urandom" >> /usr/lib/jvm/jre/lib/security/java.security \
    && yum clean all

RUN yum install -y libgomp libstdc++ && \
    rm -rf /var/cache/apk/*

ENV JAVA_HOME /etc/alternatives/jre

# Add run script as /deployments/run-java.sh and make it executable
COPY run-java.sh /deployments/
RUN chmod 755 /deployments/run-java.sh

# Run under user "jboss" and prepare for be running
# under OpenShift, too
RUN groupadd -r jboss -g 1000 \
  && useradd -u 1000 -r -g jboss -m -d /opt/jboss -s /sbin/nologin jboss \
  && chmod 755 /opt/jboss \
  && chown -R jboss /deployments \
  && usermod -g root -G `id -g jboss` jboss \
  && chmod -R "g+rwX" /deployments \
  && chown -R jboss:root /deployments

USER jboss

ENV AB_OFF=true

ENV JAVA_OPTIONS="-Dfiles.path=./models/ -Dconfig.override_with_env_vars=true -Devaluator.tensorflow.h5_converter.path=/deployments/h5_converter"

COPY --from=base /usr/src/app/api/target/lib/* /deployments/lib/
COPY --from=base /usr/src/app/api/target/api-1.0.1-SNAPSHOT.jar /deployments/app.jar
COPY --from=python-builder /usr/src/app/dist/h5_converter /deployments/h5_converter

WORKDIR /deployments

ENTRYPOINT [ "/deployments/run-java.sh" ]
