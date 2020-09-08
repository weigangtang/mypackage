Create ssh key for GitHub

ssh-keygen -t rsa -b 4096 -C “tangw5@mcmaster.ca”



How to install package from GitHub

pip install git+ssh://git@github.com/weigangtang/mypackage#egg=mypackage

import mypackage.filetools

or

from mypackages import filetools

