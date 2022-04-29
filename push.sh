#!/bin/bash
docker login
docker tag laion_idle_cah:v0 andreaskoepf/laion:cah_v0
docker push andreaskoepf/laion:cah_v0
