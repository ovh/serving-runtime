FROM maven:3.6.1-jdk-11-slim AS base

RUN apt-get update && \
    apt-get install make libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

# Stage 1: Build the onnxruntime java library
FROM ubuntu:16.04 AS onnxruntime

ARG ONNXRUNTIME_REPO=https://github.com/Microsoft/onnxruntime
ARG ONNXRUNTIME_SERVER_BRANCH=master

RUN apt-get update &&\
    apt-get install software-properties-common -y &&\
    add-apt-repository ppa:openjdk-r/ppa -y &&\
    apt-get install -y sudo git bash openjdk-8-jre openjdk-8-jdk unzip language-pack-en

WORKDIR /code

ENV PATH /opt/miniconda/bin:/code/cmake-3.14.3-Linux-x86_64/bin:/code/gradle-6.2.2/bin:${PATH}

# Prepare onnxruntime repository & build onnxruntime
RUN git clone --single-branch --branch ${ONNXRUNTIME_SERVER_BRANCH} --recursive ${ONNXRUNTIME_REPO} onnxruntime &&\
    locale-gen en_US.UTF-8 && update-locale LANG=en_US.UTF-8 &&\
    /bin/sh onnxruntime/dockerfiles/scripts/install_common_deps.sh &&\
    wget --quiet https://services.gradle.org/distributions/gradle-6.2.2-bin.zip &&\
    unzip gradle-6.2.2-bin.zip && rm gradle-6.2.2-bin.zip &&\
    cd onnxruntime &&\
    /bin/sh ./build.sh --config RelWithDebInfo --build_shared_lib --parallel --skip_tests --build_java &&\
    mv build/Linux/RelWithDebInfo/java/build/libs/onnxruntime-1.2.0-all.jar ../onnxruntime-1.2.0-all.jar &&\
    cd .. &&\
    rm -rf onnxruntime cmake-3.14.3-Linux-x86_64 gradle-6.2.2-bin.zip

## Stage 3 : build with maven builder image
FROM base AS build

COPY . /usr/src/app/
COPY --from=onnxruntime /code/onnxruntime-1.2.0-all.jar /usr/src/app/onnxruntime-1.2.0-all.jar

RUN mvn install:install-file \
	-Dfile=onnxruntime-1.2.0-all.jar \
	-DgroupId=ai \
	-DartifactId=onnxruntime \
	-Dversion=1.2.0-all \
	-Dpackaging=jar \
	-DgeneratePom=true

RUN make build MAVEN_PROFILE=onnx

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

RUN yum install -y libgomp && \
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

ENV JAVA_OPTIONS="-Dfiles.path=./models/ -Dconfig.override_with_env_vars=true"

COPY --from=build /usr/src/app/api/target/lib/* /deployments/lib/
COPY --from=build /usr/src/app/api/target/api-1.0.1-SNAPSHOT.jar /deployments/app.jar

WORKDIR /deployments

ENTRYPOINT [ "/deployments/run-java.sh" ]