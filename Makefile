WORKDIR := $(dir $(realpath $(firstword $(MAKEFILE_LIST))))
NAME := serving-runtime-base
REGISTRY :=
REPOSITORY := infaas
.DEFAULT_GOAL := build
TAG := $(lastword $(subst /, ,$(shell git rev-parse --abbrev-ref HEAD)))
M2 := '$(HOME)/.m2'

H5_CONVERTER := h5_converter/dist/h5_converter
MAVEN_PROFILE=full

.PHONY: docker-base
docker-base:
	docker build --target base -t $(NAME) -f dockerfiles/$(MAVEN_PROFILE).Dockerfile .

.PHONY: docker-test
docker-test: docker-base
	docker run --rm -v $(WORKDIR):/usr/src/app -v $(M2):/root/.m2 $(NAME) make test H5_CONVERTER=/usr/src/bin/h5_converter

.PHONY: docker-test
docker-build: docker-base
	docker run --rm -v $(WORKDIR):/usr/src/app -v $(M2):/root/.m2 $(NAME) make build H5_CONVERTER=/usr/src/bin/h5_converter

.PHONY: docker-build-api
docker-build-api:
	docker build --build-arg MAVEN_PROFILE=$(MAVEN_PROFILE) -t $(NAME) -f dockerfiles/$(MAVEN_PROFILE).Dockerfile .

.PHONY: docker-push-api
docker-push-api:
	docker tag $(NAME) $(REGISTRY)/$(REPOSITORY)/$(NAME):$(TAG)
	docker push $(REGISTRY)/$(REPOSITORY)/$(NAME):$(TAG)

.PHONY: build
build:
	mvn package -DskipTests -B -P$(MAVEN_PROFILE)

.PHONY: test
test:
	mvn -B verify -DtrimStackTrace=false -Devaluator.tensorflow.h5_converter.path=$(H5_CONVERTER) -P$(MAVEN_PROFILE)

.PHONY: deploy
deploy:
	mvn -B deploy -DskipTests -P$(MAVEN_PROFILE)

.PHONY: initialize
initialize:
	make -C evaluator-tensorflow/h5_converter build
