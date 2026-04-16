# Default version if not provided
V ?= latest
USER = stylianouvasilis
IMG = scaling-llms

docker-build-training:
	docker build --target training -t $(IMG):training-$(V) .
	docker tag $(IMG):training-$(V) $(USER)/$(IMG):training-$(V)
	docker push $(USER)/$(IMG):training-$(V)

docker-build-dev:
	docker build --target dev -t $(IMG):dev-$(V) .
	docker tag $(IMG):dev-$(V) $(USER)/$(IMG):dev-$(V)
	docker push $(USER)/$(IMG):dev-$(V)