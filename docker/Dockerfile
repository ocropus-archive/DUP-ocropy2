FROM ocropy2-base
ARG REDO=none
ARG USER=tmb
ARG DUID=1000
ARG DGID=1000
ARG PASSWORD=nopassword
ARG SHELL=/bin/bash
MAINTAINER Tom Breuel <tmbdev@gmail.com>

ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

ADD nvidia-smi /usr/local/bin
RUN chmod 755 /usr/local/bin/nvidia-smi

EXPOSE 5900

ENV HOME=/root
ADD homedir/.vimrc /root/.vimrc

ENV USER=${USER}
ENV HOME=/home/${USER}
ENV SHELL=${SHELL}
ENV DUID=${DUID}
ENV DGID=${DGID}
RUN echo "${USER}:x:${DUID}:${DGID}:${USER},,,:${HOME}:/bin/bash" >> /etc/passwd
RUN echo "${USER}:x:${DGID}" >> /etc/group
RUN echo "${USER} ALL=(ALL:ALL) NOPASSWD:ALL" >> /etc/sudoers

RUN mkdir -p $HOME
COPY homedir $HOME/
RUN mkdir -p $HOME/.vnc
RUN echo ${PASSWORD} | vncpasswd -f > $HOME/.vnc/passwd
RUN chmod 600 $HOME/.vnc/passwd

RUN mkdir /current && chown $DUID.$DGID /current
WORKDIR /current

COPY repos $HOME/repos
RUN cd $HOME/repos/hocr-tools && python setup.py install
RUN cd $HOME/repos/ocropy && python setup.py install
RUN cd $HOME/repos/cctc && python setup.py install
RUN cd $HOME/repos/dlinputs && python setup.py install
RUN cd $HOME/repos/dlmodels && python setup.py install
RUN cd $HOME/repos/dltrainers && python setup.py install
RUN cd $HOME/repos/ocropy2 && python setup.py install
RUN rm -rf repos

ADD runvnc /usr/local/bin
RUN chmod 755 /usr/local/bin/runvnc

RUN chown -R ${DUID}.${DGID} ${HOME}

# CMD /usr/local/bin/runvnc
# ENTRYPOINT /usr/local/bin/runvnc
USER ${DUID}:${DGID}
