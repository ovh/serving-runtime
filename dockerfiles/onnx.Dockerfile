## Stage 1 : Build project for onnx
FROM maven:3.6.1-jdk-11-slim AS base

RUN apt-get update && \
    apt-get install make libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src/app

COPY . /usr/src/app/

RUN make build MAVEN_PROFILE=onnx

## Stage 2 : create the docker final image
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

COPY --from=base /usr/src/app/api/target/lib/* /deployments/lib/
COPY --from=base /usr/src/app/api/target/api-1.0.1-SNAPSHOT.jar /deployments/app.jar

WORKDIR /deployments

ENTRYPOINT [ "/deployments/run-java.sh" ]