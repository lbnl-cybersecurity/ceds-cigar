# Base system stuff
FROM ubuntu:18.04
RUN apt-get -y update && apt-get install -y python3 sudo curl git python3-pip libsm6 libxext6 libxrender-dev

# Force wheel installs, and do this first so docker caches it
ADD requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

# Move all files:
ADD * /
ADD pycigar /pycigar

# Get the module installed
RUN python3 setup.py develop

# Test
# RUN cd /pycigar/tests/sanity_test/ && python3 sanity_test.py
# RUN cd /pycigar/examples/multiagent/single_policy_head/ && python3 global_single_relative_discrete_ppo.py