set -a

PATH=$HOME/bin:$PATH
PATH="/sbin:/usr/sbin:/usr/local/bin:/usr/bin:/bin"
PATH=".:$PATH"

PATH=$PATH:"/usr/local/cuda/bin"
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/nvvm/lib64:/usr/local/cuda/
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/nvidia/lib64

EDITOR=vim
VISUAL=$EDITOR

LANG=C
LC_CTYPE=en_US.utf-8
PYTHONIOENCODING=utf-8
# SUDO_ASKPASS=/usr/bin/ssh-askpass

uid=$(id -u)
PS1="@${HOSTNAME:-$(hostname)}:"'\W\$ '

set +a

alias ls="LC_COLLATE=C /bin/ls -hBX -I '*.o' -I '*.pyc' -I '*.retry' --color=auto --group-directories-first"
alias ll="ls -l"
alias ll="ls -lt | more"
