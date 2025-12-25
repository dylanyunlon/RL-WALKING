#!/bin/bash
ps -ef | grep danserver | awk '{print $2}' | xargs kill -9
ps -ef | grep danzero | awk '{print $2}' | xargs kill -9
ps -ef | grep client | awk '{print $2}' | xargs kill -9
ps -ef | grep actor | awk '{print $2}' | xargs kill -9