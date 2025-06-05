#!/usr/bin/env bash

coverage run suite.py --test test_numfolio/
coverage report -m
coverage html
