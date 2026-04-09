# Default version if not provided
V ?= latest
USER = stylianouvasilis
IMG = scaling-llms

docker-build:
	docker build -t $(IMG):$(V) .
	docker tag $(IMG):$(V) $(USER)/$(IMG):$(V)
	docker push $(USER)/$(IMG):$(V)