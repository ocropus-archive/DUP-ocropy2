define gitget
	(test -d $(2) || git clone $(1) $(2)) || (cd $(2) && git pull)
endef

image:
	docker build -t ocropy2-base base
	-mkdir -p repos
	-mkdir -p homedir/.vnc
	echo ${DOCKER_VNC:-nopassword} | vncpasswd -f > homedir/.vnc/passwd
	$(call gitget,https://github.com/tmbdev/hocr-tools.git,repos/hocr-tools)
	$(call gitget,https://github.com/tmbdev/ocropy.git,repos/ocropy)
	$(call gitget,https://github.com/tmbdev/cctc.git,repos/cctc)
	$(call gitget,https://github.com/NVLabs/dlinputs.git,repos/dlinputs)
	$(call gitget,https://github.com/NVLabs/dlmodels.git,repos/dlmodels)
	$(call gitget,https://github.com/NVLabs/dltrainers.git,repos/dltrainers)
	$(call gitget,git@github.com:tmbdev/ocropy2.git,repos/ocropy2)
	test -f repos/ocropy2/ocr-default.pt || curl https://storage.googleapis.com/tmbmodels/ocr-default.pt > repos/ocropy2/ocr-default.pt
	docker build --no-cache -t ocropy2 .

clean:
	rm -rf repos
	rm -rf homedir/.vnc

clean-docker: clean
	-docker container prune -f
	-docker image prune -f
	-docker rmi ocropy2
	-docker rmi ocropy2-base
