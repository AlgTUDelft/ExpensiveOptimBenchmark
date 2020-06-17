# NOTE: openfoam4 is NOT supported on the latest LTS release of Ubuntu.
# See: https://openfoam.org/download/4-1-ubuntu/
# Most notably, the releases supported are:
#    14.04 LTS, codename trusty (32bit and 64bit)
#    16.04 LTS, codename xenial (64bit only)
#    16.10, codename yakkety (64bit only, added 4 Nov 2016)
#    17.04, codename zesty (64bit only, added 16 Apr 2017)

sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key | apt-key add -"
sudo add-apt-repository http://dl.openfoam.org/ubuntu
sudo apt-get update
sudo apt-get -y install openfoam4